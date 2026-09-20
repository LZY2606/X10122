//! Structured evidence for a single rate-limiting decision.
//!
//! The methods named `check` / `check_key` (and their batch variants)
//! only report whether a request was allowed. The `decide` /
//! `decide_key` methods (see [`RateLimiter`][crate::RateLimiter]) take
//! the exact same decision, but return a [`Decision`]: a snapshot of
//! *all* evidence produced by that single GCRA state transition - the
//! decision time, the number of cells that were requested, the quota
//! in effect, the remaining burst capacity after an allowed request
//! and, for a rejected request, the earliest time at which a retry
//! could conform.
//!
//! Every field of a `Decision` is computed while the rate limiter's
//! state for the (keyed) request is held; a decision call advances the
//! GCRA state exactly once (and not at all when the request is
//! rejected). Compare-and-swap retries inside a keyed state store are
//! never observable: only the evidence of the transition that actually
//! happened is returned.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # fn main() {
//! use governor::clock::FakeRelativeClock;
//! use governor::{Quota, RateLimiter};
//! use nonzero_ext::nonzero;
//!
//! let clock = FakeRelativeClock::default();
//! let lim = RateLimiter::direct_with_clock(Quota::per_second(nonzero!(50u32)), clock);
//! let decision = lim.decide();
//! assert!(decision.is_allowed());
//! assert_eq!(decision.num_cells().get(), 1);
//! assert_eq!(decision.quota().burst_size().get(), 50);
//! assert_eq!(decision.remaining_burst_capacity(), 49);
//!
//! // Exhaust the burst capacity; the next decision is rejected:
//! for _ in 0..49 {
//!     assert!(lim.decide().is_allowed());
//! }
//! let rejected = lim.decide();
//! assert!(rejected.is_rejected());
//! // A rejected request consumes no cells:
//! assert!(lim.decide().is_rejected());
//! // ... but it still reports when a retry might succeed:
//! assert_eq!(rejected.wait_time_from(rejected.decided_at()), core::time::Duration::from_millis(20));
//! # }
//! # #[cfg(not(feature = "std"))]
//! # fn main() {}
//! ```

use core::fmt;
use core::num::NonZeroU32;
use core::time::Duration;

use crate::{clock, middleware::StateSnapshot, NotUntil, Quota};

/// The structured outcome of a single rate-limiting decision.
///
/// A `Decision` is produced by the `decide` family of methods on
/// [`RateLimiter`][crate::RateLimiter]. It records everything that is
/// known about a request at the exact instant it was judged:
///
/// * When the decision was made ([`Decision::decided_at`]).
/// * How many cells the request asked for
///   ([`Decision::num_cells`]).
/// * Which [`Quota`] was used ([`Decision::quota`]).
/// * How much burst capacity remains *after* an allowed request
///   ([`Decision::remaining_burst_capacity`]). This is `0` for rejected
///   requests, which do not consume any cells.
/// * For a rejected request, the earliest instant at which the request
///   might conform ([`Decision::retry_after`]) and how long to wait
///   from a given reference time ([`Decision::wait_time_from`]).
///
/// A decision is a point-in-time snapshot: none of its fields change
/// when time passes or when other requests are judged. It implements
/// `Copy`, so it is cheap to pass along to upper layers (e.g. as the
/// basis for response headers or metrics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision<P: clock::Reference> {
    decided_at: P,
    start: P,
    num_cells: NonZeroU32,
    allowed: bool,
    snapshot: StateSnapshot,
}

impl<P: clock::Reference> Decision<P> {
    #[inline]
    pub(crate) fn allow(
        decided_at: P,
        num_cells: NonZeroU32,
        snapshot: StateSnapshot,
        start: P,
    ) -> Self {
        Self {
            decided_at,
            start,
            num_cells,
            allowed: true,
            snapshot,
        }
    }

    #[inline]
    pub(crate) fn reject(
        decided_at: P,
        num_cells: NonZeroU32,
        snapshot: StateSnapshot,
        start: P,
    ) -> Self {
        Self {
            decided_at,
            start,
            num_cells,
            allowed: false,
            snapshot,
        }
    }

    /// Returns `true` if the requested cells were allowed through.
    #[inline]
    pub fn is_allowed(&self) -> bool {
        self.allowed
    }

    /// Returns `true` if the requested cells were rejected (and no
    /// state was consumed by the request).
    #[inline]
    pub fn is_rejected(&self) -> bool {
        !self.allowed
    }

    /// Returns the instant at which the decision was made, as reported
    /// by the rate limiter's clock.
    ///
    /// All time-sensitive evidence in this decision is relative to
    /// this reading.
    #[inline]
    pub fn decided_at(&self) -> P {
        self.decided_at
    }

    /// Returns the number of cells the request asked for (1 for single
    /// cell requests).
    #[inline]
    pub fn num_cells(&self) -> NonZeroU32 {
        self.num_cells
    }

    /// Returns the [`Quota`] configuration used to reach this decision.
    #[inline]
    pub fn quota(&self) -> Quota {
        self.snapshot.quota()
    }

    /// Returns the number of cells that could be let through
    /// immediately after this decision, in addition to the cells this
    /// decision allowed.
    ///
    /// For an allowed request, this is the burst capacity remaining
    /// *after* the request's cells were deducted. For a rejected
    /// request this is `0`, because a rejection does not consume any
    /// cells.
    #[inline]
    pub fn remaining_burst_capacity(&self) -> u32 {
        if self.allowed {
            self.snapshot.remaining_burst_capacity()
        } else {
            0
        }
    }

    /// Returns the earliest instant at which the rejected request
    /// could conform, or `None` if the request was allowed.
    ///
    /// This is an absolute instant measured on the rate limiter's
    /// clock.
    #[inline]
    pub fn retry_after(&self) -> Option<P> {
        if self.allowed {
            None
        } else {
            Some(self.not_until().earliest_possible())
        }
    }

    /// Returns the minimum time that has to pass from `from` for the
    /// rejected request to be able to conform.
    ///
    /// Returns a zero [`Duration`] for allowed requests, and also when
    /// the earliest conforming time is already in the past relative to
    /// `from` (e.g. because a custom clock stood still or went
    /// backwards).
    #[inline]
    pub fn wait_time_from(&self, from: P) -> Duration {
        if self.allowed {
            Duration::from_nanos(0)
        } else {
            self.not_until().wait_time_from(from)
        }
    }

    /// The negative outcome (`NotUntil`) reconstructed from this decision's
    /// snapshot. Only called for rejected decisions.
    #[inline]
    fn not_until(&self) -> NotUntil<P> {
        NotUntil::new(self.snapshot, self.start)
    }
}

impl<P: clock::Reference> fmt::Display for Decision<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.allowed {
            write!(
                f,
                "allowed {} cell(s), {} remaining",
                self.num_cells.get(),
                self.remaining_burst_capacity()
            )
        } else {
            write!(
                f,
                "rejected {} cell(s), retry at {:?}",
                self.num_cells.get(),
                self.retry_after()
                    .expect("rejected decision has retry time"),
            )
        }
    }
}
