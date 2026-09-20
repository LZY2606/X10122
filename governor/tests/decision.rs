#![cfg(feature = "std")]

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use futures_executor::block_on;
use governor::clock::{Clock, FakeRelativeClock};
use governor::middleware::StateInformationMiddleware;
use governor::nanos::Nanos;
use governor::{InsufficientCapacity, Quota, RateLimiter};
use nonzero_ext::nonzero;

/// A controllable clock that can be advanced, stalled and rewound at will.
#[derive(Clone, Default)]
struct RewindClock {
    now: Arc<AtomicU64>,
}

impl RewindClock {
    fn set(&self, nanos: u64) {
        self.now.store(nanos, Ordering::SeqCst);
    }
}

impl Clock for RewindClock {
    type Instant = Nanos;

    fn now(&self) -> Self::Instant {
        self.now.load(Ordering::SeqCst).into()
    }
}

#[test]
fn direct_decision_allowed_fields() {
    let clock = FakeRelativeClock::default();
    let quota = Quota::per_second(nonzero!(2u32)).allow_burst(nonzero!(4u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    let decision = lim.check_with_decision();
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(1u32));
    assert_eq!(decision.quota(), quota);
    assert_eq!(decision.remaining_burst_capacity(), Some(3));
    assert_eq!(decision.retry_at(), None);
    assert_eq!(decision.time_of_decision(), clock.now());
    assert_eq!(decision.wait_time_from(clock.now()), Duration::from_nanos(0));

    clock.advance(Duration::from_millis(1));
    let decision = lim.check_with_decision();
    assert!(decision.is_allowed());
    assert_eq!(decision.remaining_burst_capacity(), Some(2));
    assert_eq!(decision.time_of_decision(), clock.now());
}

#[test]
fn direct_decision_denied_fields_and_no_capacity_consumed() {
    let clock = FakeRelativeClock::default();
    let quota = Quota::per_second(nonzero!(1u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    assert!(lim.check_with_decision().is_allowed());

    let now = clock.now();
    let decision = lim.check_with_decision();
    assert!(!decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(1u32));
    assert_eq!(decision.quota(), quota);
    assert_eq!(decision.remaining_burst_capacity(), None);
    assert_eq!(decision.time_of_decision(), now);
    assert_eq!(decision.retry_at(), Some(now + Nanos::from(1_000_000_000u64)));
    assert_eq!(decision.wait_time_from(now), Duration::from_secs(1));

    // A denied decision must not consume capacity: exactly one cell is
    // available at the advertised retry time, not fewer and not more.
    clock.advance(Duration::from_secs(1));
    assert!(lim.check_with_decision().is_allowed());
    assert!(!lim.check_with_decision().is_allowed());
}

#[test]
fn direct_decision_batch() {
    let clock = FakeRelativeClock::default();
    let quota = Quota::per_second(nonzero!(1u32)).allow_burst(nonzero!(4u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    let decision = lim.check_n_with_decision(nonzero!(3u32)).unwrap();
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(3u32));
    assert_eq!(decision.quota(), quota);
    assert_eq!(decision.remaining_burst_capacity(), Some(1));
    assert_eq!(decision.time_of_decision(), clock.now());

    // Only one cell of burst capacity remains, so a batch of two is denied:
    let decision = lim.check_n_with_decision(nonzero!(2u32)).unwrap();
    assert!(!decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(2u32));
    assert!(decision.retry_at().is_some());

    // A batch larger than the burst size can never conform; the error
    // boundary is identical to check_n's:
    assert_eq!(
        lim.check_n_with_decision(nonzero!(5u32)),
        Err(InsufficientCapacity(4))
    );
    assert_eq!(lim.check_n(nonzero!(5u32)), Err(InsufficientCapacity(4)));

    // The failed batches above consumed nothing:
    let decision = lim.check_n_with_decision(nonzero!(1u32)).unwrap();
    assert!(decision.is_allowed());
    assert_eq!(decision.remaining_burst_capacity(), Some(0));
}

#[test]
fn keyed_decision_per_key() {
    let clock = FakeRelativeClock::default();
    let quota = Quota::per_second(nonzero!(1u32)).allow_burst(nonzero!(2u32));
    let lim = RateLimiter::dashmap_with_clock(quota, clock.clone());

    let decision = lim.check_key_with_decision(&"one");
    assert!(decision.is_allowed());
    assert_eq!(decision.remaining_burst_capacity(), Some(1));
    assert_eq!(decision.quota(), quota);

    // Keys have independent states:
    let decision = lim.check_key_with_decision(&"two");
    assert!(decision.is_allowed());
    assert_eq!(decision.remaining_burst_capacity(), Some(1));

    let decision = lim.check_key_n_with_decision(&"one", nonzero!(2u32)).unwrap();
    assert!(!decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(2u32));
    let retry_at = decision.retry_at().unwrap();
    assert!(retry_at > clock.now());

    // ...while "two" still has capacity left:
    let decision = lim.check_key_with_decision(&"two");
    assert!(decision.is_allowed());
    assert_eq!(decision.remaining_burst_capacity(), Some(0));
}

#[test]
fn decision_remains_self_consistent_when_clock_rewinds() {
    let clock = RewindClock::default();
    clock.set(1_000_000_000); // start the limiter at t=1s
    let quota = Quota::per_second(nonzero!(1u32));
    let lim = RateLimiter::direct_with_clock(quota, clock.clone());

    assert!(lim.check_with_decision().is_allowed());

    // The clock jumps backwards by 500ms; the decision's timestamps and
    // wait times must still agree with each other:
    clock.set(500_000_000);
    let decision = lim.check_with_decision();
    assert!(!decision.is_allowed());
    assert_eq!(decision.time_of_decision(), Nanos::from(500_000_000u64));
    let retry_at = decision.retry_at().unwrap();
    assert_eq!(retry_at, Nanos::from(2_000_000_000u64));
    assert_eq!(
        decision.wait_time_from(clock.now()),
        Duration::from_millis(1500)
    );

    // A stalled clock keeps reporting the same, consistent wait:
    let stalled = lim.check_with_decision();
    assert_eq!(stalled.retry_at(), Some(retry_at));
    assert_eq!(stalled.wait_time_from(clock.now()), Duration::from_millis(1500));

    // Once the clock reaches the advertised retry time, the cell conforms:
    clock.set(2_000_000_000);
    assert!(lim.check_with_decision().is_allowed());
}

#[test]
fn keyed_decision_concurrent_same_key() {
    const BURST: u32 = 32;
    const THREADS: u32 = 8;

    let clock = FakeRelativeClock::default();
    let quota = Quota::per_second(nonzero!(1u32)).allow_burst(nonzero!(BURST));
    let lim = Arc::new(RateLimiter::dashmap_with_clock(quota, clock));
    let barrier = Arc::new(Barrier::new(THREADS as usize));

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let lim = Arc::clone(&lim);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut decisions = Vec::new();
                barrier.wait();
                for _ in 0..BURST {
                    decisions.push(lim.check_key_with_decision(&7u32));
                }
                decisions
            })
        })
        .collect();
    let decisions: Vec<_> = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();

    // With a frozen clock, exactly the burst capacity can be allowed through,
    // no matter how the CAS retries interleaved:
    let allowed: Vec<_> = decisions.iter().filter(|d| d.is_allowed()).collect();
    assert_eq!(allowed.len(), BURST as usize);

    // Each positive decision observed the state transition it caused: every
    // remaining-capacity value occurs exactly once across all threads.
    let mut remainings: Vec<u32> = allowed
        .iter()
        .map(|d| d.remaining_burst_capacity().unwrap())
        .collect();
    remainings.sort_unstable();
    assert_eq!(remainings, (0..BURST).collect::<Vec<_>>());

    // Every denied decision carries a retry time and the same quota:
    let denied: Vec<_> = decisions.iter().filter(|d| !d.is_allowed()).collect();
    assert_eq!(denied.len(), (BURST * (THREADS - 1)) as usize);
    for decision in denied {
        assert!(decision.retry_at().is_some());
        assert_eq!(decision.quota(), quota);
        assert_eq!(decision.cells(), nonzero!(1u32));
    }
}

#[test]
fn decisions_match_legacy_check_api() {
    let quota = Quota::per_second(nonzero!(2u32)).allow_burst(nonzero!(3u32));
    let clock_old = FakeRelativeClock::default();
    let clock_new = FakeRelativeClock::default();
    let old = RateLimiter::direct_with_clock(quota, clock_old.clone())
        .with_middleware::<StateInformationMiddleware>();
    let new = RateLimiter::direct_with_clock(quota, clock_new.clone());

    // A pseudo-random but deterministic sequence of clock advances (ms):
    let steps = [0u64, 100, 100, 700, 100, 2000, 0, 300, 0, 0, 1500];
    for step in steps {
        clock_old.advance(Duration::from_millis(step));
        clock_new.advance(Duration::from_millis(step));

        let legacy = old.check();
        let decision = new.check_with_decision();

        // The allow/deny sequence is identical:
        assert_eq!(legacy.is_ok(), decision.is_allowed());
        assert_eq!(decision.time_of_decision(), clock_new.now());
        assert_eq!(decision.cells(), nonzero!(1u32));
        match legacy {
            Ok(snapshot) => {
                // The structured fields are a snapshot of the same transition:
                assert_eq!(
                    decision.remaining_burst_capacity(),
                    Some(snapshot.remaining_burst_capacity())
                );
                assert_eq!(decision.quota(), snapshot.quota());
            }
            Err(not_until) => {
                assert_eq!(decision.retry_at(), Some(not_until.earliest_possible()));
                assert_eq!(
                    decision.wait_time_from(clock_new.now()),
                    not_until.wait_time_from(clock_old.now())
                );
                assert_eq!(decision.quota(), not_until.quota());
            }
        }
    }
}

#[test]
fn batch_decisions_match_legacy_check_n_api() {
    let quota = Quota::per_second(nonzero!(1u32)).allow_burst(nonzero!(3u32));
    let clock_old = FakeRelativeClock::default();
    let clock_new = FakeRelativeClock::default();
    let old = RateLimiter::direct_with_clock(quota, clock_old.clone())
        .with_middleware::<StateInformationMiddleware>();
    let new = RateLimiter::direct_with_clock(quota, clock_new.clone());

    let attempts: Vec<(u64, NonZeroU32)> = vec![
        (0, nonzero!(2u32)),
        (0, nonzero!(2u32)),
        (1000, nonzero!(3u32)),
        (0, nonzero!(4u32)), // exceeds burst size: InsufficientCapacity
        (3000, nonzero!(1u32)),
    ];
    for (advance_ms, n) in attempts {
        clock_old.advance(Duration::from_millis(advance_ms));
        clock_new.advance(Duration::from_millis(advance_ms));

        let legacy = old.check_n(n);
        let decision = new.check_n_with_decision(n);

        match (legacy, decision) {
            (Ok(legacy), Ok(decision)) => {
                assert_eq!(legacy.is_ok(), decision.is_allowed());
                assert_eq!(decision.cells(), n);
                match legacy {
                    Ok(snapshot) => {
                        assert_eq!(
                            decision.remaining_burst_capacity(),
                            Some(snapshot.remaining_burst_capacity())
                        );
                        assert_eq!(decision.quota(), snapshot.quota());
                    }
                    Err(not_until) => {
                        assert_eq!(decision.retry_at(), Some(not_until.earliest_possible()));
                        assert_eq!(decision.quota(), not_until.quota());
                    }
                }
            }
            (Err(capacity), Err(decision_capacity)) => {
                assert_eq!(capacity, decision_capacity);
            }
            (legacy, decision) => panic!(
                "legacy and decision APIs disagree: {:?} vs {:?}",
                legacy, decision
            ),
        }
    }
}

#[test]
fn until_ready_with_decision_waits_and_returns_decision() {
    let quota = Quota::per_second(nonzero!(10u32));
    let lim = RateLimiter::direct(quota);

    // exhaust the limiter:
    loop {
        if lim.check().is_err() {
            break;
        }
    }
    let decision = block_on(lim.until_ready_with_decision());
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(1u32));
    assert_eq!(decision.quota(), quota);
    assert_eq!(decision.remaining_burst_capacity(), Some(0));
}

#[test]
fn until_n_ready_with_decision_waits_and_returns_decision() {
    let quota = Quota::per_second(nonzero!(10u32));
    let lim = RateLimiter::direct(quota);

    for _ in 0..6 {
        lim.check().unwrap();
    }
    let decision = block_on(lim.until_n_ready_with_decision(nonzero!(5u32))).unwrap();
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(5u32));
    assert_eq!(decision.quota(), quota);

    // Batches exceeding the burst size fail exactly like check_n:
    assert_eq!(
        block_on(lim.until_n_ready_with_decision(nonzero!(11u32))),
        Err(InsufficientCapacity(10))
    );
}

#[test]
fn until_key_ready_with_decision_waits_and_returns_decision() {
    let quota = Quota::per_second(nonzero!(10u32));
    let lim = RateLimiter::keyed(quota);

    // exhaust the limiter:
    loop {
        if lim.check_key(&1u32).is_err() {
            break;
        }
    }
    let decision = block_on(lim.until_key_ready_with_decision(&1u32));
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(1u32));
    assert_eq!(decision.quota(), quota);

    let decision = block_on(lim.until_key_n_ready_with_decision(&2u32, nonzero!(3u32))).unwrap();
    assert!(decision.is_allowed());
    assert_eq!(decision.cells(), nonzero!(3u32));
}
