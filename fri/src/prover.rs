use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
//use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
//use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{ info_span, instrument}; //debug_span,

use crate::{fold_even_odd, CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<G, F, EF, M, Challenger>(
    _g: &G,
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger,
) ->  (FriProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: Field,
    EF: ExtensionField<F> + TwoAdicField,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<F, EF>,
{
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();
    let commit_phase_result = commit_phase(config, input, log_max_height, challenger);
    let pow_witness = challenger.grind(config.proof_of_work_bits);
    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| answer_query(config, &commit_phase_result.data, index))
            .collect()
    });
    
    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: F, //sp1
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<Challenge>>; 32],
    log_max_height: usize,    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    //G: FriGenericConfig<Val, Challenge>,
{
    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        let leaves = RowMajorMatrix::new(current.clone(), 2);
        //tracing::debug!("##########333333 leaves:{:?}", leaves);
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);
        data.push(prover_data);

        //according sp1-v5
        let beta: Challenge = challenger.sample_algebra_element();
        
        current = fold_even_odd(current, beta);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());
    let final_poly = current[0];
    for x in current {
        assert_eq!(x, final_poly);
    }
    
    challenger.observe_algebra_element(final_poly);
    
    //tracing::debug!("##########333333 commit_phase_commits.data:{:?}", data);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> QueryProof<F, M>
where
    F: Field,
    M: Mmcs<F>,
{
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_pair, commit).unpack();
            assert_eq!(opened_rows.len(), 1);
            let opened_row = &opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}
