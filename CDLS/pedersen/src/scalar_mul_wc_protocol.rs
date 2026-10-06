//! Without-c1-commitment-to-scalar variant of EC scalar multiplication proof.
//!
//! Uses the optimised point-addition proof [OptECPointAddProof] internally
//! (Optimisations 1, 2, 3)

use ark_ec::{
    short_weierstrass::{self as sw},
    AffineRepr, CurveConfig, CurveGroup,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use ark_std::Zero;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};

use crate::{
    ec_point_add_protocol_opt::{OptECPointAddIntermediate, OptECPointAddProof},
    pedersen_config::{PedersenComm, PedersenConfig},
    transcript::ECScalarMulTranscript,
};

type ScalarOC<P> = <<P as PedersenConfig>::OCurve as CurveConfig>::ScalarField;
type AffOC<P> = sw::Affine<<P as PedersenConfig>::OCurve>;
type ProjOC<P> = sw::Projective<<P as PedersenConfig>::OCurve>;

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct ECScalarMulWCProof<P: PedersenConfig> {
    /// c1_x, c1_y: commitments to Z' = αK.
    pub c1_x: sw::Affine<P>,
    pub c1_y: sw::Affine<P>,
    /// c2_x, c2_y: commitments to Z'' = (α-ω)K.
    pub c2_x: sw::Affine<P>,
    pub c2_y: sw::Affine<P>,

    /// alpha: response. If b=0, α = α (the random prover scalar) so αK = Z'.
    /// If b=1, α = α-ω so αK = Z''.
    pub alpha: <<P as PedersenConfig>::OCurve as CurveConfig>::ScalarField,
    pub tau_x: <P as CurveConfig>::ScalarField,
    pub tau_y: <P as CurveConfig>::ScalarField,

    /// Inner optimised point-addition proof: Z + Z'' = Z'.
    pub eap: OptECPointAddProof<P>,
}

/// Prover state for one round between the first message and the response.
pub struct ECScalarMulWCProofIntermediate<P: PedersenConfig> {
    alpha: ScalarOC<P>,
    alpha_minus_omega: ScalarOC<P>,
    z_prime: AffOC<P>,
    z_double_prime: AffOC<P>,
    c1_x: PedersenComm<P>,
    c1_y: PedersenComm<P>,
    c2_x: PedersenComm<P>,
    c2_y: PedersenComm<P>,
    eap: OptECPointAddIntermediate<P>,
}

impl<P: PedersenConfig> ECScalarMulWCProof<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn make_transcript(
        transcript: &mut Transcript,
        k: &sw::Affine<<P as PedersenConfig>::OCurve>,
        c_zx: &sw::Affine<P>,
        c_zy: &sw::Affine<P>,
        c1_x: &sw::Affine<P>,
        c1_y: &sw::Affine<P>,
        c2_x: &sw::Affine<P>,
        c2_y: &sw::Affine<P>,
    ) {
        ECScalarMulTranscript::domain_sep(transcript);

        let mut buf = Vec::new();
        k.serialize_compressed(&mut buf).unwrap();
        ECScalarMulTranscript::append_point(transcript, b"K", &buf[..]);
        buf.clear();

        for (label, pt) in [
            (b"C_Zx" as &[u8], c_zx),
            (b"C_Zy", c_zy),
            (b"C1_x", c1_x),
            (b"C1_y", c1_y),
            (b"C2_x", c2_x),
            (b"C2_y", c2_y),
        ] {
            pt.serialize_compressed(&mut buf).unwrap();
            ECScalarMulTranscript::append_point(transcript, label, &buf[..]);
            buf.clear();
        }
    }

    /// Prover step 1: all group work for one round. No transcript access,
    /// so many rounds can run in parallel.
    pub fn first_message<T: RngCore + CryptoRng>(
        rng: &mut T,
        z_pt: &AffOC<P>,                           // S = ω*K
        z_scalar: &ScalarOC<P>,                    // ω (witness)
        k: &AffOC<P>,                              // K (base)
        c_z: (&PedersenComm<P>, &PedersenComm<P>), // commitments to S.x, S.y
    ) -> ECScalarMulWCProofIntermediate<P> {
        // Sample α ∈ F_q \ {0, ω, 2ω}.
        let alpha: ScalarOC<P> = loop {
            let cand = ScalarOC::<P>::rand(rng);
            if !cand.is_zero() && cand != *z_scalar && cand != (*z_scalar + *z_scalar) {
                break cand;
            }
        };
        let alpha_minus_omega = alpha - *z_scalar;

        // Z' = α*K, Z'' = (α-ω)*K.
        let k_proj = k.into_group();
        let z_pts: Vec<AffOC<P>> =
            ProjOC::<P>::normalize_batch(&[k_proj * alpha, k_proj * alpha_minus_omega]);
        let z_prime = z_pts[0];
        let z_double_prime = z_pts[1];

        // Commit to Z', Z'' coordinate-wise in P (T-curve).
        let z_prime_x_sf = <P as PedersenConfig>::from_ob_to_sf(*z_prime.x().unwrap());
        let z_prime_y_sf = <P as PedersenConfig>::from_ob_to_sf(*z_prime.y().unwrap());
        let z_dp_x_sf = <P as PedersenConfig>::from_ob_to_sf(*z_double_prime.x().unwrap());
        let z_dp_y_sf = <P as PedersenConfig>::from_ob_to_sf(*z_double_prime.y().unwrap());

        let c1_x = PedersenComm::new(z_prime_x_sf, rng);
        let c1_y = PedersenComm::new(z_prime_y_sf, rng);
        let c2_x = PedersenComm::new(z_dp_x_sf, rng);
        let c2_y = PedersenComm::new(z_dp_y_sf, rng);

        // First message of the inner optimised PA proof: S + Z'' = Z'.
        // A = S   (c_z.0, c_z.1)
        // B = Z'' (c2_x, c2_y)
        // T = Z'  (c1_x, c1_y)
        let eap = OptECPointAddProof::<P>::first_message(
            rng,
            *z_pt,          // A
            z_double_prime, // B
            c_z.0,
            c_z.1, // c1, c2 -> A
            &c2_x,
            &c2_y, // c3, c4 -> B
            &c1_x,
            &c1_y, // c5, c6 -> T
        );

        ECScalarMulWCProofIntermediate {
            alpha,
            alpha_minus_omega,
            z_prime,
            z_double_prime,
            c1_x,
            c1_y,
            c2_x,
            c2_y,
            eap,
        }
    }

    /// Derive a single challenge bit b ∈ {0, 1}. Only for the standalone
    /// single-round `create` / `verify`.
    fn challenge_bit(transcript: &mut Transcript) -> bool {
        let mut chal_buf = [0u8; 1];
        transcript.challenge_bytes(b"WC_SM_b", &mut chal_buf);
        chal_buf[0] & 1 == 1
    }

    /// Prover step 2: transcript operations for one round WITHOUT the challenge
    /// bit: inner PA absorb + challenge (a full field element, so per-round is
    /// fine), then the SM statement/commitments. The caller must derive the
    /// challenge bits only after every round has been absorbed.
    pub fn append_without_bit(
        transcript: &mut Transcript,
        k: &AffOC<P>,
        c_z: (&PedersenComm<P>, &PedersenComm<P>),
        inter: &ECScalarMulWCProofIntermediate<P>,
    ) -> <P as CurveConfig>::ScalarField {
        let eap_chal = OptECPointAddProof::<P>::append_and_challenge(
            transcript,
            c_z.0,
            c_z.1,
            &inter.c2_x,
            &inter.c2_y,
            &inter.c1_x,
            &inter.c1_y,
            &inter.eap,
        );

        Self::make_transcript(
            transcript,
            k,
            &c_z.0.comm,
            &c_z.1.comm,
            &inter.c1_x.comm,
            &inter.c1_y.comm,
            &inter.c2_x.comm,
            &inter.c2_y.comm,
        );

        eap_chal
    }

    /// Prover step 3: responses for one round. Field arithmetic only.
    pub fn respond(
        z_pt: &AffOC<P>,
        c_z: (&PedersenComm<P>, &PedersenComm<P>),
        inter: &ECScalarMulWCProofIntermediate<P>,
        eap_chal: &<P as CurveConfig>::ScalarField,
        b: bool,
    ) -> Self {
        // Verifies that ω*K + (α-ω)*K = α*K.
        let eap = OptECPointAddProof::<P>::create_proof_with_challenge(
            *z_pt,                // A
            inter.z_double_prime, // B
            inter.z_prime,        // T
            &inter.eap,
            c_z.0,
            c_z.1, // c1, c2 -> A
            &inter.c2_x,
            &inter.c2_y, // c3, c4 -> B
            &inter.c1_x,
            &inter.c1_y, // c5, c6 -> T
            eap_chal,
        );

        let (alpha, tau_x, tau_y) = if b {
            (inter.alpha_minus_omega, inter.c2_x.r, inter.c2_y.r)
        } else {
            (inter.alpha, inter.c1_x.r, inter.c1_y.r)
        };

        Self {
            c1_x: inter.c1_x.comm,
            c1_y: inter.c1_y.comm,
            c2_x: inter.c2_x.comm,
            c2_y: inter.c2_y.comm,
            alpha,
            tau_x,
            tau_y,
            eap,
        }
    }

    /// Create ONE standalone round (soundness error 1/2).
    /// WARNING: never loop this for soundness amplification; the bit is derived
    /// immediately, so a loop can be ground round by round. Use
    /// [crate::fs_scalar_mul_wc_protocol::FSECScalarMulWCProof] instead.
    pub fn create<T: RngCore + CryptoRng>(
        transcript: &mut Transcript,
        rng: &mut T,
        z_pt: &AffOC<P>,                           // S = ω*K
        z_scalar: &ScalarOC<P>,                    // ω (witness)
        k: &AffOC<P>,                              // K (base)
        c_z: (&PedersenComm<P>, &PedersenComm<P>), // commitments to S.x, S.y
    ) -> Self {
        let inter = Self::first_message(rng, z_pt, z_scalar, k, c_z);
        let eap_chal = Self::append_without_bit(transcript, k, c_z, &inter);
        let b = Self::challenge_bit(transcript);
        Self::respond(z_pt, c_z, &inter, &eap_chal, b)
    }

    /// Verifier step 1: transcript operations for one round WITHOUT the
    /// challenge bit. Returns the inner PA challenge.
    pub fn absorb_without_bit(
        &self,
        transcript: &mut Transcript,
        k: &AffOC<P>,
        c_zx: &sw::Affine<P>,
        c_zy: &sw::Affine<P>,
    ) -> <P as CurveConfig>::ScalarField {
        let eap_chal = self.eap.absorb(
            transcript, c_zx, c_zy, // c1, c2 -> A = S
            &self.c2_x, &self.c2_y, // c3, c4 -> B = Z''
            &self.c1_x, &self.c1_y, // c5, c6 -> T = Z'
        );

        Self::make_transcript(
            transcript, k, c_zx, c_zy, &self.c1_x, &self.c1_y, &self.c2_x, &self.c2_y,
        );

        eap_chal
    }

    /// Verifier step 2: all checks for one round given its challenges. No
    /// transcript access, so many rounds can be checked in parallel. Only call
    /// with challenges derived from the transcript.
    pub fn verify_with_challenge(
        &self,
        k: &AffOC<P>,
        c_zx: &sw::Affine<P>,
        c_zy: &sw::Affine<P>,
        eap_chal: &<P as CurveConfig>::ScalarField,
        b: bool,
    ) -> bool {
        let pa_ok = self.eap.verify_with_challenge(
            c_zx, c_zy, // c1, c2 -> A = S
            &self.c2_x, &self.c2_y, // c3, c4 -> B = Z''
            &self.c1_x, &self.c1_y, // c5, c6 -> T = Z'
            eap_chal,
        );
        if !pa_ok {
            return false;
        }

        // αK = Z' (b=0) or Z'' (b=1). Verifier reconstructs from α and public K.
        let alpha_k = (k.into_group() * self.alpha).into_affine();
        let alpha_k_x_sf = <P as PedersenConfig>::from_ob_to_sf(*alpha_k.x().unwrap());
        let alpha_k_y_sf = <P as PedersenConfig>::from_ob_to_sf(*alpha_k.y().unwrap());

        let expected_x_p = P::msm_generators(&alpha_k_x_sf, &self.tau_x);
        let expected_y_p = P::msm_generators(&alpha_k_y_sf, &self.tau_y);
        let expected = sw::Projective::<P>::normalize_batch(&[expected_x_p, expected_y_p]);

        if b {
            expected[0] == self.c2_x && expected[1] == self.c2_y
        } else {
            expected[0] == self.c1_x && expected[1] == self.c1_y
        }
    }

    /// Verify ONE standalone round (soundness error 1/2).
    /// WARNING: never loop this for soundness amplification; use
    /// [crate::fs_scalar_mul_wc_protocol::FSECScalarMulWCProof] instead.
    pub fn verify(
        &self,
        transcript: &mut Transcript,
        k: &AffOC<P>,
        c_zx: &sw::Affine<P>,
        c_zy: &sw::Affine<P>,
    ) -> bool {
        let eap_chal = self.absorb_without_bit(transcript, k, c_zx, c_zy);
        let b = Self::challenge_bit(transcript);
        self.verify_with_challenge(k, c_zx, c_zy, &eap_chal, b)
    }
}
