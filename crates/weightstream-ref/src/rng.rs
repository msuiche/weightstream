//! Bit-exact port of the two numpy pieces the Engram layout depends on:
//! `np.random.default_rng(seed)` (SeedSequence -> PCG64 XSL-RR 128/64) and
//! `Generator.integers(low, high, size, dtype=int64)` (Lemire's bounded
//! method). The trained hash multipliers were produced by exactly this
//! stream (`engram.py:64-83`), so the reference engine must reproduce it
//! bit-for-bit or the n-gram tables would be read at wrong rows.
//!
//! Verified against numpy 2.5.2 for seeds 10007*{1,3,14,39} and several
//! vocab sizes (see `tests::numpy_rng_multipliers_match`).

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// PCG64 XSL-RR 128/64, seeded the way numpy's `default_rng(u64 seed)` does.
pub struct NumpyPcg64 {
    state: u128,
    inc: u128,
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value;
    v ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> XSHIFT;
    v
}

fn mix32(x: u32, y: u32) -> u32 {
    let mut r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^= r >> XSHIFT;
    r
}

/// `numpy.random.SeedSequence(seed).pool` (pool_size 4).
fn seed_sequence_pool(seed: u64) -> [u32; 4] {
    // _int_to_uint32_array: base-2^32 digits, lowest word first
    let mut entropy: Vec<u32> = Vec::new();
    let mut rest = seed;
    if rest == 0 {
        entropy.push(0);
    }
    while rest > 0 {
        entropy.push(rest as u32);
        rest >>= 32;
    }

    let mut pool = [0u32; 4];
    let mut hash_const = INIT_A;
    for i in 0..4 {
        pool[i] = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                pool[i_dst] = mix32(pool[i_dst], hashmix(pool[i_src], &mut hash_const));
            }
        }
    }
    for i_src in 4..entropy.len() {
        for i_dst in 0..4 {
            pool[i_dst] = mix32(pool[i_dst], hashmix(entropy[i_src], &mut hash_const));
        }
    }
    pool
}

/// `SeedSequence.generate_state(4, uint64)`.
fn generate_state(pool: &[u32; 4], n_words: usize) -> Vec<u64> {
    let mut hash_const = INIT_B;
    let mut words = Vec::with_capacity(n_words * 2);
    for i in 0..n_words * 2 {
        let mut d = pool[i % 4];
        d ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        d = d.wrapping_mul(hash_const);
        d ^= d >> XSHIFT;
        words.push(d);
    }
    (0..n_words)
        .map(|i| words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32)
        .collect()
}

impl NumpyPcg64 {
    /// Equivalent to `np.random.default_rng(seed)`.
    pub fn new(seed: u64) -> Self {
        let val = generate_state(&seed_sequence_pool(seed), 4);
        let initstate = (val[0] as u128) << 64 | val[1] as u128;
        let initseq = (val[2] as u128) << 64 | val[3] as u128;
        let inc = initseq << 1 | 1; // 128-bit shift; initseq < 2^128 fits
                                    // pcg_setseq_128_srandom_r: state=0; step; state += initstate; step
                                    // pcg_setseq_128_srandom_r: state=0; step; state += initstate; step
        let state = 0u128.wrapping_mul(PCG_MULT).wrapping_add(inc); // = inc
        let state = state.wrapping_add(initstate);
        let state = state.wrapping_mul(PCG_MULT).wrapping_add(inc);
        Self { state, inc }
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// One raw 64-bit draw (XSL-RR output).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        let v = hi ^ lo;
        let r = (hi >> 58) as u32;
        v.rotate_right(r)
    }

    /// `Generator.integers(0, bound, n, dtype=int64)` — Lemire's method with
    /// the inclusive upper bound `bound - 1` (numpy `_rand_int64` path).
    /// `bound` must be in `[1, 2^63]` (the engram multiplier bound obeys).
    pub fn integers(&mut self, bound: u64, n: usize) -> Vec<u64> {
        assert!(bound >= 1 && bound <= 1 << 63);
        let rng = bound - 1; // inclusive upper bound
        let rng_excl = bound;
        let threshold = (u64::MAX - rng) % rng_excl;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            loop {
                let x = self.next_u64();
                let m = (x as u128) * (rng_excl as u128);
                let leftover = m as u64;
                if leftover >= rng_excl {
                    out.push((m >> 64) as u64);
                    break;
                }
                if leftover >= threshold {
                    out.push((m >> 64) as u64);
                    break;
                }
            }
        }
        out
    }
}

/// splitmix64 — cheap deterministic stream for synthetic weights and the
/// Gumbel sampler (NOT the numpy path).
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D4_9BB1_3311_3EB1);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_rng_multipliers_match() {
        // Values captured from numpy 2.5.2:
        //   (np.random.default_rng(10007*layer).integers(0, bound, 4) * 2 + 1)
        // with bound = max(1, (np.iinfo(np.int64).max // cv) // 2).
        let cases: &[(u32, u32, [u64; 4])] = &[
            (
                1,
                99092,
                [
                    76632096046245,
                    4839876093313,
                    35959672319349,
                    73987337458391,
                ],
            ),
            (
                14,
                99092,
                [
                    67716810739261,
                    51510806800915,
                    30921347202721,
                    82619226485591,
                ],
            ),
            (
                3,
                32,
                [
                    85240722207167499,
                    252868047889091175,
                    33684873675973757,
                    217683273637190441,
                ],
            ),
            (
                39,
                16,
                [
                    487925436340976507,
                    413436462867417603,
                    334446646158292295,
                    519253381846724255,
                ],
            ),
        ];
        for &(layer, cv, want) in cases {
            let bound = (((1u64 << 63) - 1) / cv as u64) / 2;
            let mut rng = NumpyPcg64::new(10007 * layer as u64);
            let got: Vec<u64> = rng
                .integers(bound, 4)
                .into_iter()
                .map(|v| v * 2 + 1)
                .collect();
            assert_eq!(got.as_slice(), &want[..], "layer {layer} cv {cv}");
        }
    }

    #[test]
    fn splitmix64_known() {
        let mut s = 0u64;
        let a = splitmix64(&mut s);
        let b = splitmix64(&mut s);
        assert_ne!(a, b);
        assert_eq!(s, 0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(2));
    }
}
