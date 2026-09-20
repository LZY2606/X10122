//! Behavior tests for the structured `Decision` interface (`decide` family).
//!
//! These tests use a controllable clock or thread barriers, never timing-based
//! sleeps, so the synchronous behavior is deterministic.

use core::num::NonZeroU64;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use governor::clock::Clock;
use governor::nanos::Nanos;
use governor::state::StateStore;
use governor::{InsufficientCapacity, Quota, RateLimiter};
use nonzero_ext::nonzero;

/// A controllable clock whose reading can be set absolutely, so it can stand
/// still and move backwards as well as forwards.
#[derive(Clone, Default, Debug)]
struct ManualClock(Arc<std::sync::atomic::AtomicU64>);

impl ManualClock {
    fn set(&self, nanos: u64) {
        self.0.store(nanos, Ordering::SeqCst);
    }
    fn advance(&self, by: Duration) {
        self.0.fetch_add(by.as_nanos() as u64, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    type Instant = Nanos;
    fn now(&self) -> Nanos {
        Nanos::from(self.0.load(Ordering::SeqCst))
    }
}

const SECOND: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Direct, single cell
// ---------------------------------------------------------------------------

#[test]
fn direct_single_cell_evidence() {
    let clock = ManualClock::default();
    clock.set(1_000_000_000);
    let quota = Quota::per_second(nonzero!(5u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    let d = lim.decide();
    assert!(d.is_allowed());
    assert!(!d.is_rejected());
    assert_eq!(d.decided_at(), Nanos::from(1_000_000_000));
    assert_eq!(d.num_cells(), nonzero!(1u32));
    assert_eq!(d.quota(), quota);
    assert_eq!(d.remaining_burst_capacity(), 4);
    assert_eq!(d.retry_after(), None);
    assert_eq!(d.wait_time_from(d.decided_at()), Duration::ZERO);
}

#[test]
fn direct_decision_drains_burst_in_order() {
    let clock = ManualClock::default();
    // 100ms per cell, burst 3: replenishment math lands on clean instants.
    let quota = Quota::with_period(Duration::from_millis(100))
        .unwrap()
        .allow_burst(nonzero!(3u32));
    let lim = RateLimiter::direct_with_clock(quota, clock);

    for remaining in [2u32, 1, 0] {
        let d = lim.decide();
        assert!(d.is_allowed(), "a burst slot should be allowed");
        assert_eq!(d.remaining_burst_capacity(), remaining);
    }

    // 4th decision at the same frozen instant is rejected without consuming.
    let rejected = lim.decide();
    assert!(rejected.is_rejected());
    assert_eq!(rejected.remaining_burst_capacity(), 0);
    assert_eq!(rejected.num_cells(), nonzero!(1u32));
    // The first cell replenishes one interval after the first allowed cell (t=0).
    assert_eq!(rejected.retry_after(), Some(Nanos::from(100_000_000)));
    assert_eq!(
        rejected.wait_time_from(Nanos::from(0)),
        Duration::from_millis(100)
    );
}

// ---------------------------------------------------------------------------
// Batch cells
// ---------------------------------------------------------------------------

#[test]
fn direct_batch_consumes_all_cells() {
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(5u32)), clock);

    let d = lim.decide_n(nonzero!(3u32)).unwrap();
    assert!(d.is_allowed());
    assert_eq!(d.num_cells(), nonzero!(3u32));
    assert_eq!(d.remaining_burst_capacity(), 2);

    // exactly the remaining 2 fit:
    assert!(lim.decide_n(nonzero!(2u32)).unwrap().is_allowed());
    assert!(lim.decide_n(nonzero!(1u32)).unwrap().is_rejected());
}

#[test]
fn direct_batch_rejected_consumes_nothing() {
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(5u32)), clock);

    assert!(lim.decide_n(nonzero!(3u32)).unwrap().is_allowed());
    // 3 more do not fit (only 2 remain): rejected, nothing consumed.
    let rejected = lim.decide_n(nonzero!(3u32)).unwrap();
    assert!(rejected.is_rejected());
    assert_eq!(rejected.num_cells(), nonzero!(3u32));
    // the 2 that remained still fit afterwards:
    assert!(lim.decide_n(nonzero!(2u32)).unwrap().is_allowed());
    assert!(lim.decide().is_rejected());
}

#[test]
fn direct_batch_beyond_burst_matches_check_n_boundary() {
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(4u32)), clock.clone());

    // burst size 4 => n = 5 can never conform; identical error from both APIs.
    assert_eq!(
        lim.decide_n(nonzero!(5u32)).unwrap_err(),
        lim.check_n(nonzero!(5u32)).unwrap_err(),
    );
    assert_eq!(
        lim.decide_n(nonzero!(5u32)).unwrap_err(),
        InsufficientCapacity(4)
    );
    // the capacity check happens before any state change:
    assert!(lim.decide_n(nonzero!(4u32)).unwrap().is_allowed());
}

// ---------------------------------------------------------------------------
// Keyed (single + batch)
// ---------------------------------------------------------------------------

#[test]
fn keyed_single_and_batch_evidence() {
    let clock = ManualClock::default();
    let lim = RateLimiter::hashmap_with_clock(Quota::per_second(nonzero!(4u32)), clock);

    let d = lim.decide_key(&"a");
    assert!(d.is_allowed());
    assert_eq!(d.remaining_burst_capacity(), 3);

    let batch = lim.decide_key_n(&"a", nonzero!(2u32)).unwrap();
    assert!(batch.is_allowed());
    assert_eq!(batch.remaining_burst_capacity(), 1);

    assert!(lim.decide_key(&"a").is_allowed());
    let rejected = lim.decide_key(&"a");
    assert!(rejected.is_rejected());
    assert!(rejected.retry_after().is_some());

    // a different key has its own fresh budget:
    assert_eq!(lim.decide_key(&"b").remaining_burst_capacity(), 3);
}

#[test]
fn keyed_batch_beyond_burst_is_insufficient_capacity() {
    let clock = ManualClock::default();
    let lim = RateLimiter::hashmap_with_clock(Quota::per_second(nonzero!(2u32)), clock);
    assert_eq!(
        lim.decide_key_n(&"k", nonzero!(3u32)).unwrap_err(),
        InsufficientCapacity(2),
    );
    // state untouched:
    assert!(lim.decide_key_n(&"k", nonzero!(2u32)).unwrap().is_allowed());
}

// ---------------------------------------------------------------------------
// Compare-and-swap retries: failed attempts must not appear in the evidence
// ---------------------------------------------------------------------------

/// A direct (single-key) state store whose first closure evaluation is always
/// fed stale, fully-replenished state and whose result is discarded, exactly
/// like a compare-and-swap attempt that loses to a concurrent writer. It
/// records that a failed attempt happened.
#[derive(Clone, Default)]
struct FlakyStateStore {
    state: Arc<portable_atomic::AtomicU64>,
    forced_failures: Arc<AtomicUsize>,
}

impl StateStore for FlakyStateStore {
    type Key = ();

    fn measure_and_replace<T, F, E>(&self, _key: &(), mut f: F) -> Result<T, E>
    where
        F: FnMut(Option<Nanos>) -> Result<(T, Nanos), E>,
    {
        // Phase 1: a doomed attempt. It is fed stale, fully-replenished state
        // and its outcome is thrown away, just like a compare-and-swap that
        // loses to a concurrent writer. This phase never lands.
        if self
            .forced_failures
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let _ = f(Some(Nanos::from(0)));
        }

        // Phase 2: the real attempt runs against the actual state and lands.
        let prev = self.state.load(Ordering::Acquire);
        let observed = NonZeroU64::new(prev).map(|n| Nanos::from(n.get()));
        match f(observed) {
            Ok((result, next)) => {
                self.state.store(u64::from(next), Ordering::Release);
                Ok(result)
            }
            Err(e) => Err(e),
        }
    }
}

#[test]
fn failed_cas_attempt_is_not_part_of_the_evidence() {
    // Construct at t=0 so the limiter's reference instant is 0.
    let clock = ManualClock::default();
    let store = FlakyStateStore::default();
    let lim: governor::RateLimiter<
        (),
        FlakyStateStore,
        ManualClock,
        governor::middleware::NoOpMiddleware<Nanos>,
    > = RateLimiter::new(
        Quota::per_second(nonzero!(5u32)),
        store.clone(),
        clock.clone(),
    );

    // A "concurrent writer" leaves behind a real theoretical arrival time 0.8s
    // in the past, while the clock is at t=1s. The doomed first attempt is fed
    // a different stale value (no state at all) by the store.
    clock.set(SECOND.as_nanos() as u64);
    store.state.store(200_000_000, Ordering::SeqCst);

    let d = lim.decide_key(&());
    // A stale attempt must have been made and discarded...
    assert_eq!(store.forced_failures.load(Ordering::SeqCst), 1);
    // ...yet the evidence is computed from the real state (TAT = 0.2s): the
    // decision is allowed and the full burst is available (real TAT is 0.8s in
    // the past, well beyond the 0.8s burst window), giving 4 remaining cells.
    assert!(d.is_allowed());
    assert_eq!(d.remaining_burst_capacity(), 4);
    assert_eq!(d.decided_at(), Nanos::from(SECOND.as_nanos() as u64));
    // The real transition lands at max(t0, real TAT) + one 200ms cell = 1.2s.
    // The discarded stale attempt would only ever have produced 0.2s.
    assert_eq!(store.state.load(Ordering::SeqCst), 1_200_000_000);
}

// ---------------------------------------------------------------------------
// Rejections do not consume quota
// ---------------------------------------------------------------------------

#[test]
fn rejection_does_not_consume_quota() {
    let clock = ManualClock::default();
    let quota = Quota::per_second(nonzero!(1u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    assert!(lim.decide().is_allowed());

    // Any number of rejected decisions at the frozen instant leave the state
    // untouched, and they all report the same retry time.
    for _ in 0..100 {
        let d = lim.decide();
        assert!(d.is_rejected());
        assert_eq!(d.remaining_burst_capacity(), 0);
        assert_eq!(d.retry_after(), Some(Nanos::from(SECOND.as_nanos() as u64)));
    }

    // Advance exactly to the earliest retry time: exactly one cell is available.
    clock.advance(SECOND);
    let d = lim.decide();
    assert!(d.is_allowed());
    assert_eq!(d.remaining_burst_capacity(), 0);
    assert!(lim.decide().is_rejected());
}

// ---------------------------------------------------------------------------
// Custom clocks that stand still or move backwards
// ---------------------------------------------------------------------------

#[test]
fn stopped_clock_evidence_is_self_consistent() {
    let clock = ManualClock::default();
    clock.set(5 * SECOND.as_nanos() as u64);
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(1u32)), clock);

    let d = lim.decide();
    assert!(d.is_allowed());
    // The clock never advances: the retry instant is in the future relative to
    // the (frozen) decision time, and the reported wait time bridges exactly
    // that gap.
    let rejected = lim.decide();
    assert!(rejected.is_rejected());
    let decided_at = rejected.decided_at();
    let retry = rejected.retry_after().unwrap();
    assert_eq!(rejected.wait_time_from(decided_at), SECOND);
    assert_eq!(decided_at + rejected.wait_time_from(decided_at), retry);
    // Asking the wait from a time already at/past the retry instant saturates.
    assert_eq!(rejected.wait_time_from(retry), Duration::ZERO);
}

#[test]
fn backwards_clock_evidence_is_self_consistent() {
    let clock = ManualClock::default();
    let quota = Quota::per_second(nonzero!(1u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    // Consume the burst at t=10s.
    clock.set(10 * SECOND.as_nanos() as u64);
    assert!(lim.decide().is_allowed());

    // Clock jumps backwards to t=1s. The decision is rejected, and its time,
    // wait and retry fields stay mutually consistent via saturating arithmetic.
    clock.set(SECOND.as_nanos() as u64);
    let d = lim.decide();
    assert!(d.is_rejected());
    assert_eq!(d.decided_at(), Nanos::from(SECOND.as_nanos() as u64));
    let retry = d.retry_after().unwrap();
    // TAT is 11s (10s decision + 1s per cell); from the rolled-back t=1s the
    // wait is 10s even though the clock moved backwards.
    assert_eq!(retry, Nanos::from(11 * SECOND.as_nanos() as u64));
    assert_eq!(d.wait_time_from(d.decided_at()), 10 * SECOND);
    assert_eq!(d.decided_at() + d.wait_time_from(d.decided_at()), retry);

    // Jump all the way back to (and before) the limiter's start instant: the
    // wait computation stays non-negative and still bridges to the retry time.
    clock.set(0);
    let d = lim.decide();
    assert!(d.is_rejected());
    assert_eq!(d.wait_time_from(d.decided_at()), 11 * SECOND);
    assert_eq!(
        d.retry_after(),
        Some(Nanos::from(11 * SECOND.as_nanos() as u64))
    );

    // A reference point already at/past the retry instant always yields a zero
    // wait (saturating subtraction), even with a pathological clock.
    assert_eq!(
        d.wait_time_from(Nanos::from(11 * SECOND.as_nanos() as u64)),
        Duration::ZERO
    );
}

// ---------------------------------------------------------------------------
// High concurrency on the same key: exactly one state transition per call
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
#[test]
fn high_concurrency_same_key_single_cell() {
    use crossbeam;

    const BURST: u32 = 100;
    const THREADS: usize = 16;

    let clock = ManualClock::default(); // frozen: no replenishment
    let lim =
        RateLimiter::hashmap_with_clock(Quota::per_second(NonZeroU32::new(BURST).unwrap()), clock);

    let allowed = Arc::new(AtomicUsize::new(0));
    let rejected = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(THREADS));

    crossbeam::scope(|scope| {
        for _ in 0..THREADS {
            let lim = &lim;
            let allowed = Arc::clone(&allowed);
            let rejected = Arc::clone(&rejected);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move |_| {
                barrier.wait();
                // Each thread makes more attempts than the total burst, so we
                // are guaranteed to cover both allowed and rejected outcomes.
                for _ in 0..20 {
                    let d = lim.decide_key(&"k");
                    if d.is_allowed() {
                        assert_eq!(d.num_cells(), nonzero!(1u32));
                        allowed.fetch_add(1, Ordering::SeqCst);
                    } else {
                        assert_eq!(d.remaining_burst_capacity(), 0);
                        rejected.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    })
    .unwrap();

    // Exactly BURST cells ever got through at the frozen instant.
    assert_eq!(allowed.load(Ordering::SeqCst), BURST as usize);
    assert!(rejected.load(Ordering::SeqCst) > 0);
}

#[cfg(feature = "std")]
#[test]
fn high_concurrency_same_key_evidence_matches_transitions() {
    use crossbeam;

    // Track every accepted remaining-capacity value using a shared set, to
    // prove evidence is drawn only from transitions that actually happened:
    // each value 0..BURST occurs exactly once.
    const BURST: u32 = 50;
    const THREADS: usize = 8;

    let clock = ManualClock::default();
    let lim =
        RateLimiter::hashmap_with_clock(Quota::per_second(NonZeroU32::new(BURST).unwrap()), clock);
    let counts = Arc::new(std::sync::Mutex::new([0usize; BURST as usize]));
    let rejected = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(THREADS));

    crossbeam::scope(|scope| {
        for _ in 0..THREADS {
            let lim = &lim;
            let counts = Arc::clone(&counts);
            let rejected = Arc::clone(&rejected);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move |_| {
                barrier.wait();
                for _ in 0..20 {
                    let d = lim.decide_key(&"k");
                    if d.is_allowed() {
                        let rem = d.remaining_burst_capacity() as usize;
                        counts.lock().unwrap()[rem] += 1;
                    } else {
                        rejected.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    })
    .unwrap();

    let counts = counts.lock().unwrap();
    for (rem, &count) in counts.iter().enumerate() {
        assert_eq!(
            count, 1,
            "remaining capacity {rem} must be observed exactly once"
        );
    }
    assert!(rejected.load(Ordering::SeqCst) > 0);
}

#[cfg(all(feature = "std", feature = "dashmap"))]
#[test]
fn high_concurrency_dashmap_same_key() {
    use crossbeam;

    const BURST: u32 = 64;
    const THREADS: usize = 12;

    let clock = ManualClock::default();
    let lim =
        RateLimiter::dashmap_with_clock(Quota::per_second(NonZeroU32::new(BURST).unwrap()), clock);
    let allowed = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(std::sync::Barrier::new(THREADS));

    crossbeam::scope(|scope| {
        for _ in 0..THREADS {
            let lim = &lim;
            let allowed = Arc::clone(&allowed);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move |_| {
                barrier.wait();
                for _ in 0..20 {
                    if lim.decide_key(&"k").is_allowed() {
                        allowed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    })
    .unwrap();

    assert_eq!(allowed.load(Ordering::SeqCst), BURST as usize);
}

// ---------------------------------------------------------------------------
// Consistency: the old `check` API and the new `decide` API take identical
// decisions; the structured fields are just a snapshot of the same transition.
// ---------------------------------------------------------------------------

/// Replay the same deterministic clock script on two limiters, one driven by
/// the old API and one by the new API, and assert their admit sequences match.
#[test]
fn old_and_new_interfaces_agree_direct() {
    use governor::middleware::StateInformationMiddleware;
    use governor::RateLimiter as RL;

    let quota = Quota::per_second(nonzero!(5u32));

    let old_clock = ManualClock::default();
    let new_clock = old_clock.clone();
    let old_lim = RL::direct_with_clock(quota, old_clock.clone())
        .with_middleware::<StateInformationMiddleware>();
    let new_lim = RL::direct_with_clock(quota, new_clock);

    // A script of (single-cell call, batch size, clock advance in ms).
    let script: &[(bool, u32, u64)] = &[
        (true, 1, 0),
        (false, 3, 0),
        (true, 1, 50),
        (false, 2, 0),
        (true, 1, 300),
        (false, 6, 0), // exceeds burst: InsufficientCapacity on both
        (false, 4, 0),
        (true, 1, 1_000),
        (false, 5, 0),
        (true, 1, 0),
    ];

    for &(single, n, advance_ms) in script {
        old_clock.advance(Duration::from_millis(advance_ms));
        if single {
            let old = old_lim.check();
            let new = new_lim.decide();
            assert_eq!(old.is_ok(), new.is_allowed(), "single-cell disagreement");
            if let Ok(snapshot) = old {
                assert_eq!(
                    snapshot.remaining_burst_capacity(),
                    new.remaining_burst_capacity(),
                );
                assert_eq!(snapshot.quota(), new.quota());
            }
        } else {
            let n = NonZeroU32::new(n).unwrap();
            let old = old_lim.check_n(n);
            let new = new_lim.decide_n(n);
            assert_eq!(
                old.is_err(),
                new.is_err(),
                "InsufficientCapacity must agree"
            );
            if let (Ok(old), Ok(new)) = (old, new) {
                assert_eq!(old.is_ok(), new.is_allowed());
                if let Ok(snapshot) = old {
                    assert_eq!(
                        snapshot.remaining_burst_capacity(),
                        new.remaining_burst_capacity(),
                    );
                }
            }
        }
    }
}

#[test]
fn old_and_new_interfaces_agree_keyed() {
    use governor::middleware::StateInformationMiddleware;
    use governor::RateLimiter as RL;

    let quota = Quota::per_second(nonzero!(4u32));
    let old_clock = ManualClock::default();
    let new_clock = old_clock.clone();
    let old_lim = RL::hashmap_with_clock(quota, old_clock.clone())
        .with_middleware::<StateInformationMiddleware>();
    let new_lim = RL::hashmap_with_clock(quota, new_clock);

    let keys = [1u32, 2, 1, 2, 1];
    let batches = [1u32, 2, 3, 1, 5];
    let advances_ms = [0u64, 100, 0, 250, 1_000];

    for (&key, (&n, &advance_ms)) in keys.iter().zip(batches.iter().zip(advances_ms.iter())) {
        old_clock.advance(Duration::from_millis(advance_ms));
        let n = NonZeroU32::new(n).unwrap();
        let old = old_lim.check_key_n(&key, n);
        let new = new_lim.decide_key_n(&key, n);
        assert_eq!(
            old.is_err(),
            new.is_err(),
            "InsufficientCapacity must agree"
        );
        if let (Ok(old), Ok(new)) = (old, new) {
            assert_eq!(old.is_ok(), new.is_allowed(), "admission disagreement");
            if let Ok(snapshot) = old {
                assert_eq!(
                    snapshot.remaining_burst_capacity(),
                    new.remaining_burst_capacity(),
                );
                assert_eq!(snapshot.quota(), new.quota());
                assert_eq!(new.num_cells(), n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Async waiting paths: same result semantics as the synchronous decisions.
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
mod async_tests {
    use super::*;
    use futures_executor::block_on;

    #[test]
    fn until_decision_ready_returns_allowed_snapshot() {
        let lim = RateLimiter::direct(Quota::per_second(nonzero!(50u32)));
        let d = block_on(lim.until_decision_ready());
        assert!(d.is_allowed());
        assert_eq!(d.num_cells(), nonzero!(1u32));
        assert_eq!(d.remaining_burst_capacity(), 49);
    }

    #[test]
    fn until_n_decision_ready_returns_allowed_snapshot() {
        let lim = RateLimiter::direct(Quota::per_second(nonzero!(50u32)));
        let d = block_on(lim.until_n_decision_ready(nonzero!(10u32))).unwrap();
        assert!(d.is_allowed());
        assert_eq!(d.num_cells(), nonzero!(10u32));
        assert_eq!(d.remaining_burst_capacity(), 40);
    }

    #[test]
    fn until_n_decision_ready_rejects_oversized_batch() {
        let lim = RateLimiter::direct(Quota::per_second(nonzero!(5u32)));
        let err = block_on(lim.until_n_decision_ready(nonzero!(6u32))).unwrap_err();
        assert_eq!(err, InsufficientCapacity(5));
        // nothing was consumed by the oversized request:
        let d = block_on(lim.until_n_decision_ready(nonzero!(5u32))).unwrap();
        assert!(d.is_allowed());
    }

    #[test]
    fn until_key_decision_ready_returns_allowed_snapshot() {
        let lim = RateLimiter::keyed(Quota::per_second(nonzero!(50u32)));
        let d = block_on(lim.until_key_decision_ready(&"k"));
        assert!(d.is_allowed());
        assert_eq!(d.remaining_burst_capacity(), 49);
    }

    #[test]
    fn until_key_n_decision_ready_returns_allowed_snapshot() {
        let lim = RateLimiter::keyed(Quota::per_second(nonzero!(50u32)));
        let d = block_on(lim.until_key_n_decision_ready(&"k", nonzero!(10u32))).unwrap();
        assert!(d.is_allowed());
        assert_eq!(d.num_cells(), nonzero!(10u32));
        assert_eq!(d.remaining_burst_capacity(), 40);
    }

    /// The waiting path returns the decision of the transition that finally
    /// allows the request. Uses a fast replenishing quota so the real wait is
    /// brief but still has a deterministic lower bound (no timing luck is
    /// asserted beyond that).
    #[test]
    fn until_decision_ready_waits_then_allows() {
        let lim = RateLimiter::direct(Quota::per_second(nonzero!(100u32)));
        // Exhaust the burst synchronously.
        while lim.check().is_ok() {}
        let start = std::time::Instant::now();
        let d = block_on(lim.until_decision_ready());
        assert!(start.elapsed() >= Duration::from_millis(9));
        assert!(d.is_allowed());
        assert_eq!(d.num_cells(), nonzero!(1u32));
    }

    #[test]
    fn until_key_decision_ready_waits_then_allows() {
        let lim = RateLimiter::keyed(Quota::per_second(nonzero!(100u32)));
        while lim.check_key(&"k").is_ok() {}
        let start = std::time::Instant::now();
        let d = block_on(lim.until_key_decision_ready(&"k"));
        assert!(start.elapsed() >= Duration::from_millis(9));
        assert!(d.is_allowed());
    }
}

// ---------------------------------------------------------------------------
// Misc: Decision display and coverage
// ---------------------------------------------------------------------------

#[test]
fn decision_display() {
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(1u32)), clock);
    let allowed = lim.decide();
    assert!(format!("{allowed}").contains("allowed 1 cell(s)"));
    let rejected = lim.decide();
    let shown = format!("{rejected}");
    assert!(shown.contains("rejected 1 cell(s)"), "got: {}", shown);
    assert!(shown.contains("retry at"));
}

#[test]
fn decision_equality_and_copy() {
    let clock = ManualClock::default();
    let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(2u32)), clock);
    let d1 = lim.decide();
    let d2 = d1; // Copy
    assert_eq!(d1, d2);
    assert_eq!(d1.remaining_burst_capacity(), d2.remaining_burst_capacity());
}
