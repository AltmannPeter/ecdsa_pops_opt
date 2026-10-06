//! FS-wrapped WC variant of EC scalar multiplication proof.
//! Repeats the single-round WC protocol SECPARAM times for binary-challenge soundness.

use ark_ec::{
    short_weierstrass::{self as sw},
    CurveConfig,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use rand::rngs::StdRng;
use rand::{CryptoRng, RngCore, SeedableRng};
use rayon::prelude::*;

use crate::{
    pedersen_config::{PedersenComm, PedersenConfig},
    scalar_mul_wc_protocol::ECScalarMulWCProof,
    transcript::ECScalarMulTranscript,
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct FSECScalarMulWCProof<P: PedersenConfig> {
    proofs: Vec<ECScalarMulWCProof<P>>,
}

impl<P: PedersenConfig> FSECScalarMulWCProof<P> {
    /// Derive all SECPARAM challenge bits at once. Must only be called after
    /// every round has been absorbed into the transcript.
    fn challenge_bits(transcript: &mut Transcript) -> Vec<bool> {
        let mut buf = vec![0u8; (P::SECPARAM + 7) / 8];
        transcript.challenge_bytes(b"WC_SM_bits", &mut buf);
        (0..P::SECPARAM).map(|i| (buf[i / 8] >> (i % 8)) & 1 == 1).collect()
    }

    /// Create. Runs the WC protocol SECPARAM times with independent randomness;
    /// all challenge bits are derived after all rounds are absorbed.
    pub fn create<T: RngCore + CryptoRng>(
        transcript: &mut Transcript,
        rng: &mut T,
        z_pt: &sw::Affine<<P as PedersenConfig>::OCurve>,
        z_scalar: &<<P as PedersenConfig>::OCurve as CurveConfig>::ScalarField,
        k: &sw::Affine<<P as PedersenConfig>::OCurve>,
        c_z: (&PedersenComm<P>, &PedersenComm<P>),
    ) -> Self {
        ECScalarMulTranscript::domain_sep(transcript);

        // Independent per-round RNGs, seeded from the caller's CSPRNG.
        // Every round MUST get fresh masks: reusing α across two rounds with
        // different bits would reveal both α and α-ω, hence ω.
        let seeds: Vec<[u8; 32]> = (0..P::SECPARAM)
            .map(|_| {
                let mut s = [0u8; 32];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();

        // 1. First messages (all group work), in parallel.
        let inters: Vec<_> = seeds
            .par_iter()
            .map(|seed| {
                let mut prng = StdRng::from_seed(*seed);
                ECScalarMulWCProof::<P>::first_message(&mut prng, z_pt, z_scalar, k, c_z)
            })
            .collect();

        // 2. Absorb every round's first message, THEN derive all bits at once.
        let eap_chals: Vec<_> = inters
            .iter()
            .map(|inter| ECScalarMulWCProof::<P>::append_without_bit(transcript, k, c_z, inter))
            .collect();
        let bits = Self::challenge_bits(transcript);

        // 3. Responses (field arithmetic only), in parallel.
        let proofs = inters
            .par_iter()
            .zip(eap_chals.par_iter())
            .zip(bits.par_iter())
            .map(|((inter, eap_chal), b)| {
                ECScalarMulWCProof::<P>::respond(z_pt, c_z, inter, eap_chal, *b)
            })
            .collect();

        Self { proofs }
    }

    /// Verify. Replays every round into the transcript, derives all bits at
    /// once, then checks every round in parallel.
    pub fn verify(
        &self,
        transcript: &mut Transcript,
        k: &sw::Affine<<P as PedersenConfig>::OCurve>,
        c_zx: &sw::Affine<P>,
        c_zy: &sw::Affine<P>,
    ) -> bool {
        ECScalarMulTranscript::domain_sep(transcript);
        if self.proofs.len() != P::SECPARAM {
            return false;
        }

        // 1. Replay every round's first message, THEN derive all bits at once.
        let eap_chals: Vec<_> = self
            .proofs
            .iter()
            .map(|p| p.absorb_without_bit(transcript, k, c_zx, c_zy))
            .collect();
        let bits = Self::challenge_bits(transcript);

        // 2. Check all rounds in parallel.
        self.proofs
            .par_iter()
            .zip(eap_chals.par_iter())
            .zip(bits.par_iter())
            .all(|((p, eap_chal), b)| p.verify_with_challenge(k, c_zx, c_zy, eap_chal, *b))
    }
}
