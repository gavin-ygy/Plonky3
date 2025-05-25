use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, Mmcs, OpenedValues, Pcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedValue, PackedFieldExtension, TwoAdicField, PrimeCharacteristicRing, batch_multiplicative_inverse,
    cyclic_subgroup_coset_known_order, dot_product,
};
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::verifier::{self, FriError};
use crate::{FriConfig, FriGenericConfig, FriProof, prover};

#[derive(Debug)]
pub struct TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    // degree bound
    pub(crate) dft: Dft,      //sp1 :Radix2DitParallel
    pub(crate) mmcs: InputMmcs, //sp1:FieldMerkleTreeMmcs
    pub(crate) fri: FriConfig<FriMmcs>, //sp1: ExtensionMmcs<Val, Challenge, ValMmcs>;
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriConfig<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }

    pub fn fri_config(&self) -> &FriConfig<FriMmcs> {
        &self.fri
    }
}

#[derive(Debug)]
pub enum VerificationError<InputMmcsError, FriMmcsError> {
    InputMmcsError(InputMmcsError),
    FriError(FriError<FriMmcsError, ()>),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct TwoAdicFriPcsProof<
    Val: Field,
    Challenge: Field,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
> {
    pub fri_proof: FriProof<Challenge, FriMmcs, Val>,
    /// For each query, for each committed batch, query openings for that batch
    pub query_openings: Vec<Vec<BatchOpening<Val, InputMmcs>>>,
}


pub struct TwoAdicFriGenericConfig<InputProof, InputError>(
    pub PhantomData<(InputProof, InputError)>,
);

pub type TwoAdicFriGenericConfigForMmcs<F, M> =
    TwoAdicFriGenericConfig<Vec<BatchOpening<F, M>>, <M as Mmcs<F>>::Error>;

impl<F: TwoAdicField, InputProof, InputError: Debug, EF: ExtensionField<F>> FriGenericConfig<F, EF>
    for TwoAdicFriGenericConfig<InputProof, InputError>
{
    type InputProof = InputProof;
    type InputError = InputError;

    fn extra_query_index_bits(&self) -> usize {
        0
    }

    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
    ) -> EF {
        let arity = 2;
        let log_arity = 1;
        let (e0, e1) = evals
            .collect_tuple()
            .expect("TwoAdicFriFolder only supports arity=2");
        // If performance critical, make this API stateful to avoid this
        // This is a bit more math than is necessary, but leaving it here
        // in case we want higher arity in the future.
        let subgroup_start = F::two_adic_generator(log_height + log_arity)
            .exp_u64(reverse_bits_len(index, log_height) as u64);
        let mut xs = F::two_adic_generator(log_arity)
            .shifted_powers(subgroup_start)
            .take(arity)
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        assert_eq!(log_arity, 1, "can only interpolate two points for now");
        // interpolate and evaluate at beta
        e0 + (beta - xs[0]) * (e1 - e0) * (xs[1] - xs[0]).inverse()
        // Currently Algebra<F> does not include division so we do it manually.
        // Note we do not want to do an EF division as that is far more expensive.
    }

    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, m: M) -> Vec<EF> {
        // We use the fact that
        //     p_e(x^2) = (p(x) + p(-x)) / 2
        //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
        // that is,
        //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
        //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
        // so
        //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
        //
        // As p_e, p_o will be in the extension field we want to find ways to avoid extension multiplications.
        // We should only need a single one (namely multiplication by beta).
        let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();

        // TODO: vectorize this (after we have packed extension fields)

        // As beta is in the extension field, we want to avoid multiplying by it
        // for as long as possible. Here we precompute the powers  `g_inv^i / 2` in the base field.
        let mut halve_inv_powers = g_inv
            .shifted_powers(F::ONE.halve())
            .take(m.height())
            .collect_vec();
        reverse_slice_index_bits(&mut halve_inv_powers);

        m.par_rows()
            .zip(halve_inv_powers)
            .map(|(mut row, halve_inv_power)| {
                let (lo, hi) = row.next_tuple().unwrap();
                (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
            })
            .collect()
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
    <InputMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>: Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixView<'a, Val>>;
 
    type Proof = TwoAdicFriPcsProof<Val, Challenge, InputMmcs, FriMmcs>;
    type Error = VerificationError<InputMmcs::Error, FriMmcs::Error>;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        // This panics if (and only if) `degree` is not a power of 2 or `degree`
        // > `1 << Val::TWO_ADICITY`.
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::GENERATOR / domain.shift();
                // Commit to the bit-reversed LDE.
                self.dft
                    .coset_lde_batch(evals, self.fri.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        // todo: handle extrapolation for LDEs we don't have
        assert_eq!(domain.shift(), Val::GENERATOR);
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        /*

        A quick rundown of the optimizations in this function:
        We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
        for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
        in the columns of the matrices. y is computed by barycentric interpolation.
        X and p(X) are in the base field; alpha, y and z are in the extension.
        The primary goal is to minimize extension multiplications.

        - Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
        of a matrix, then multiply by an "alpha offset" when accumulating.
              a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
            = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
            (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

        - For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
        Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
            (see `inv_denoms`)

        - Then, for each matrix (with columns p_i) and opening point z, we want:
            for each row (corresponding to subgroup element X):
                reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

            We can factor out inv_denom, and expand what's left:
                reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

            And separate the sum:
                reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ] ]

            And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
            So the hot loop (that depends on both X and i) is just:
                sum_i [ alpha^i * p_i[X] ]

            with alpha^i an extension, p_i[X] a base

        */

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = self
                    .mmcs
                    .get_matrices(data)
                    .into_iter()
                    .map(|m| m.as_view())
                    .collect_vec();
                debug_assert_eq!(
                    mats.len(),
                    points.len(),
                    "each matrix should have a corresponding set of evaluation points"
                );
                (mats, points)
            })
            .collect_vec();

        // Find the maximum height and the maximum width of matrices in the batch.
        // These do not need to correspond to the same matrix.
        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No Matrices Supplied?");
        let log_global_max_height = log2_strict_usize(global_max_height);

        let mut coset = cyclic_subgroup_coset_known_order(
            Val::two_adic_generator(log_global_max_height),
            Val::GENERATOR,
            global_max_height,
        )
        .collect_vec();
        reverse_slice_index_bits(&mut coset);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(z - X) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, &coset);

        // Evaluate coset representations and write openings to the challenger
        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height() >> self.fri.log_blowup;
                        // `subgroup` and `mat` are both in bit-reversed order, so we can truncate.
                        let (low_coset, _) = mat.split_rows(h);
                        let coset_h = &coset[..h];

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                let _guard =
                                    info_span!("evaluate matrix", dims = %mat.dimensions())
                                        .entered();

                                // Use Barycentric interpolation to evaluate the matrix at the given point.
                                let ys =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            let inv_denoms = &inv_denoms.get(&point).unwrap()[..h];
                                            interpolate_coset_with_precomputation(
                                                &low_coset,
                                                Val::GENERATOR,
                                                point,
                                                coset_h,
                                                inv_denoms,
                                            )
                                        });
                                ys.iter()
                                    .for_each(|&y| challenger.observe_algebra_element(y));
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Batch combination challenge
        // TODO: Should we be computing a different alpha for each height?
        let alpha: Challenge = challenger.sample_algebra_element();

        // We precompute powers of alpha as we need the same powers for each matrix.
        // We compute both a vector of unpacked powers and a vector of packed powers.
        // TODO: It should be possible to refactor this to only use the packed powers but
        // this is not a bottleneck so is not a priority.
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // Now that we have sent the openings to the verifier, it remains to prove
        // that those openings are correct.

        // Given a low degree polynomial `f(x)` with claimed evaluation `f(zeta)`, we can check
        // that `f(zeta)` is correct by doing a low degree test on `(f(zeta) - f(x))/(zeta - x)`.
        // We will use `alpha` to batch together both different claimed openings `zeta` and
        // different polynomials `f` whose evaluation vectors have the same height.

        // TODO: If we allow different polynomials to have different blow_up factors
        // we may need to revisit this and to ensure it is safe to batch them together.

        // num_reduced records the number of reduced function opening point pairs
        // of each given `log_height`.
        let mut num_reduced = [0; 32];

        // For each `log_height` from 2^1 -> 2^32, reduced_openings will contain either `None`
        // if there are no matrices of that height, or `Some(vec)` where `vec` is equal to
        // a sum of `(f(zeta) - f(x))/(zeta - x)` over all `f`'s of that height and
        // opening points `zeta` with the sum weighted by powers of alpha.
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);

        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let _guard =
                    info_span!("reduce matrix quotient", dims = %mat.dimensions()).entered();

                let log_height = log2_strict_usize(mat.height());

                // If this is our first matrix at this height, initialise reduced_openings to zero.
                // Otherwise, get a mutable reference to it.
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![Challenge::ZERO; mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                // Treating our matrix M as the evaluations of functions M0, M1, ...
                // Compute the evaluations of `Mred(x) = M0(x) + alpha*M1(x) + ...`
                let mat_compressed = info_span!("compress mat").in_scope(|| {
                    // This will be reused for all points z which M is opened at so we collect into a vector.
                    mat.rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
                        .collect::<Vec<_>>()
                });

                for (&point, openings) in points_for_mat.iter().zip(openings_for_mat) {
                    // If we have multiple matrices at the same height, we need to scale mat to combine them.
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);

                    // As we have all the openings `Mi(z)`, we can combine them using `alpha`
                    // in an identical way to before to also compute `Mred(z)`.
                    let reduced_openings: Challenge =
                        dot_product(alpha_powers.iter().copied(), openings.iter().copied());

                    mat_compressed
                        .par_iter()
                        .zip(reduced_opening_for_log_height.par_iter_mut())
                        // inv_denoms contains `1/(point - x)` for `x` in a coset `gK`.
                        // If `|K| =/= mat.height()` we actually want a subset of this
                        // corresponding to the evaluations over `gH` for `|H| = mat.height()`.
                        // As inv_denoms is bit reversed, the evaluations over `gH` are exactly
                        // the evaluations over `gK` at the indices `0..mat.height()`.
                        // So zip will truncate to the desired smaller length.
                        .zip(inv_denoms.get(&point).unwrap().par_iter())
                        // Map the function `Mred(x) -> (Mred(z) - Mred(x))/(z - x)`
                        // across the evaluations vector of `Mred(x)`.
                        .for_each(|((&reduced_row, ro), &inv_denom)| {
                            *ro += alpha_pow_offset * (reduced_openings - reduced_row) * inv_denom
                        });
                    num_reduced[log_height] += mat.width();
                }
                
            }
        }

        //let fri_input = reduced_openings.into_iter().rev().flatten().collect_vec();

        let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
            TwoAdicFriGenericConfig(PhantomData);

        let (fri_proof, query_indices) = prover::prove(&g, &self.fri, &reduced_openings, challenger);
        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        let log_max_height = log2_strict_usize(self.mmcs.get_max_height(data));
                        let bits_reduced = log_global_max_height - log_max_height;
                        let reduced_index = index >> bits_reduced;
                        let (opened_values, opening_proof) =
                            self.mmcs.open_batch(reduced_index, data).unpack();
                        BatchOpening {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect()
            })
            .collect();

        (
            all_opened_values,
            TwoAdicFriPcsProof {
                fri_proof,
                query_openings,
            },
        )
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Write evaluations to challenger.
        for (_, round) in rounds.iter() {
            for (_, mat) in round.iter() {
                for (_, point) in mat.iter() {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }
        // Batch combination challenge
        let alpha: Challenge = challenger.sample_algebra_element();

        let fri_challenges =
            verifier::verify_shape_and_sample_challenges(&self.fri, &proof.fri_proof, challenger)
                .map_err(VerificationError::FriError)?;

        let log_global_max_height =
            proof.fri_proof.commit_phase_commits.len() + self.fri.log_blowup;

        let reduced_openings: Vec<[Challenge; 32]> = proof
            .query_openings
            .iter()
            .zip(&fri_challenges.query_indices)
            .map(|(query_opening, &index)| {
                let mut ro = [Challenge::ZERO; 32];
                let mut alpha_pow = [Challenge::ONE; 32];

                for (batch_opening, (batch_commit, mats)) in izip!(query_opening, &rounds) {
                    let batch_heights = mats
                        .iter()
                        .map(|(domain, _)| domain.size() << self.fri.log_blowup)
                        .collect_vec();
                    let batch_dims = batch_heights
                        .iter()
                        // TODO: MMCS doesn't really need width; we put 0 for now.
                        .map(|&height| Dimensions { width: 0, height })
                        .collect_vec();

                    let batch_max_height = batch_heights.iter().max().expect("Empty batch?");
                    let log_batch_max_height = log2_strict_usize(*batch_max_height);
                    let bits_reduced = log_global_max_height - log_batch_max_height;
                    let reduced_index = index >> bits_reduced;

                    self.mmcs.verify_batch(
                        batch_commit,
                        &batch_dims,
                        reduced_index,
                        batch_opening.into(),
                    )?;
                    for (mat_opening, (mat_domain, mat_points_and_values)) in
                        izip!(&batch_opening.opened_values, mats)
                    {
                        let log_height = log2_strict_usize(mat_domain.size()) + self.fri.log_blowup;

                        let bits_reduced = log_global_max_height - log_height;
                        let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

                        let x = Val::GENERATOR
                            * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                        for (z, ps_at_z) in mat_points_and_values {
                            for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                                let quotient = (-p_at_z + p_at_x) / (-*z + x);
                                ro[log_height] += alpha_pow[log_height] * quotient;
                                alpha_pow[log_height] *= alpha;
                            }
                        }
                    }
                }
                Ok(ro)
            })
            .collect::<Result<Vec<_>, InputMmcs::Error>>()
            .map_err(VerificationError::InputMmcsError)?;

        verifier::verify_challenges(
            &self.fri,
            &proof.fri_proof,
            &fri_challenges,
            &reduced_openings,
        )
        .map_err(VerificationError::FriError)?;

        Ok(())
    }
}

#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &coset[..(1 << log_height)]
                        .iter()
                        .map(|&x| z - x)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}

//sp1 
pub struct PowersReducer<F: Field, EF> {
    pub powers: Vec<EF>,
    // If EF::D = 2 and powers is [01 23 45 67],
    // this holds [[02 46] [13 57]]
    pub transposed_packed: Vec<Vec<F::Packing>>,
}

impl<F: Field, EF: ExtensionField<F>> PowersReducer<F, EF> {
    pub fn new(base: EF, max_width: usize) -> Self {
        let powers: Vec<EF> = base
            .powers()
            .take(max_width.next_multiple_of(F::Packing::WIDTH))
            .collect();

        let transposed_packed: Vec<Vec<F::Packing>> = transpose_vec(
            (0..EF::DIMENSION)
                .map(|d| {
                    F::Packing::pack_slice(
                        //&powers.iter().map(|a| a.as_base_slice()[d]).collect_vec(),
                        &powers.iter().map(|a| a.as_basis_coefficients_slice()[d]).collect_vec(),
                    )
                    .to_vec()
                })
                .collect(),
        );

        Self {
            powers,
            transposed_packed,
        }
    }

    // Compute sum_i base^i * x_i
    pub fn reduce_ext(&self, xs: &[EF]) -> EF {
        self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()
    }

    // Same as `self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()`
    pub fn reduce_base(&self, xs: &[F]) -> EF {
        let (xs_packed, xs_sfx) = F::Packing::pack_slice_with_suffix(xs);
        let mut sums = (0..EF::DIMENSION).map(|_| F::Packing::ZERO).collect::<Vec<_>>();
        for (&x, pows) in izip!(xs_packed, &self.transposed_packed) {
            for d in 0..EF::DIMENSION {
                sums[d] += x * pows[d];
            }
        }
        let packed_sum = EF::from_basis_coefficients_fn(|d| sums[d].as_slice().iter().copied().sum());//old ver: from_base_fn
        let sfx_sum = xs_sfx
            .iter()
            .zip(&self.powers[(xs_packed.len() * F::Packing::WIDTH)..])
            .map(|(&x, &pow)| pow * x)
            .sum::<EF>();
        packed_sum + sfx_sum
    }
}

fn transpose_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    //use p3_field::FieldExtensionAlgebra;
    use p3_field::BasedVectorSpace;
    use rand::{thread_rng, Rng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_powers_reducer() {
        let mut rng = rand::thread_rng();
        //let alpha: EF = rng.gen();
        let alpha: EF = rng.r#gen::<EF>();
        let n = 1000;
        let sizes = [5, 110, 512, 999, 1000];
        let r = PowersReducer::<F, EF>::new(alpha, n);

        // check reduce_ext
        for size in sizes {
            let xs: Vec<EF> = (0..size).map(|_| rng.r#gen::<EF>()).collect();
            assert_eq!(
                r.reduce_ext(&xs),
                xs.iter()
                    .enumerate()
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * x)
                    .sum()
            );
        }

        // check reduce_base
        for size in sizes {
            let xs: Vec<F> = (0..size).map(|_| rng.r#gen::<F>()).collect();
            assert_eq!(
                r.reduce_base(&xs),
                xs.iter()
                    .enumerate()
                    //.map(|(i, &x)| alpha.exp_u64(i as u64) * EF::from_base(x)) 
                    .map(|(i, &x)| alpha.exp_u64(i as u64) * x)
                    .sum()
            );
        }

        // bench reduce_base
        /*
        use core::hint::black_box;
        use std::time::Instant;
        let samples = 1_000;
        for i in 0..5 {
            let xs: Vec<F> = (0..999).map(|_| rng.gen()).collect();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base_slow(black_box(&xs)));
            }
            let dt_slow = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..samples {
                black_box(r.reduce_base(black_box(&xs)));
            }
            let dt_fast = t0.elapsed();
            println!("sample {i}: slow: {dt_slow:?} fast: {dt_fast:?}");
        }
        */
    }
}

