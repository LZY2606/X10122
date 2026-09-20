//! Structured evidence about a single rate-limiting decision.
//!
//! The [`RateLimiter`][crate::RateLimiter] `check_with_decision` family of
//! methods returns a [`Decision`]: a snapshot of everything there is to know
//! about one rate-limiting verdict, taken at the exact moment the
//! rate-limiter state transitioned (or, for negative outcomes, was observed).
//!
//! Unlike calling [`RateLimiter::check`][crate::RateLimiter::check] and then
//! interrogating the rate limiter separately, all the information in a
//! `Decision` is derived from the *same* state transition, so concurrent
//! decisions made by other threads can not skew the values relative to each
//! other.

use core::num::NonZeroU32;
use core::time::Duration;

use crate::{clock, Quota};

/// The structured outcome of a single rate-limiting decision.
///
/// A `Decision` is returned by the `check_with_decision` family of methods
/// (e.g. [`RateLimiter::check_with_decision`][crate::RateLimiter::check_with_decision])
/// and captures, from the single state transition that decided the fate of a
/// request:
///
/// * the time at which the decision was made
///   ([`time_of_decision`][Decision::time_of_decision]),
/// * the number of cells the decision was made for
///   ([`cells`][Decision::cells]),
/// * the quota the decision was reached under ([`quota`][Decision::quota]),
/// * and either the remaining burst capacity after a positive decision
///   ([`remaining_burst_capacity`][Decision::remaining_burst_capacity]) or
///   the earliest time a retry could be allowed after a negative one
///   ([`retry_at`][Decision::retry_at]).
///
/// # Example
/// ```rust
/// # #[cfg(feature = "std")]
/// # fn main () {
/// use governor::{Quota, RateLimiter};
/// use nonzero_ext::nonzero;
///
/// let lim = RateLimiter::direct(Quota::per_second(nonzero!(1_u32)));
///
/// // The first cell is allowed, and uses up the entire burst capacity:
/// let decision = lim.check_with_decision();
/// assert!(decision.is_allowed());
/// assert_eq!(decision.remaining_burst_capacity(), Some(0));
///
/// // The next cell is denied, and the decision says when to retry:
/// let decision = lim.check_with_decision();
/// assert!(!decision.is_allowed());
/// assert!(decision.retry_at().is_some());
/// # }
/// # #[cfg(not(feature = "std"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision<P: clock::Reference> {
    /// The quota the decision was reached under.
    quota: Quota,

    /// The number of cells the decision was made for.
    cells: NonZeroU32,

    /// The clock reading at which the decision was made.
    at: P,

    /// The outcome-specific evidence of the decision.
    outcome: Outcome<P>,
}

/// The outcome-specific part of a [`Decision`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome<P: clock::Reference> {
    /// The cells were allowed through; the burst capacity that remains
    /// immediately after the decision is recorded.
    Allowed { remaining_burst_capacity: u32 },

    /// The cells were not allowed through; the earliest time at which a
    /// retry could be allowed is recorded.
    Denied { retry_at: P },
}

impl<P: clock::Reference> Decision<P> {
    /// Creates a positive (conforming) decision.
    pub(crate) fn allowed(
        quota: Quota,
        cells: NonZeroU32,
        at: P,
        remaining_burst_capacity: u32,
    ) -> Self {
        Decision {
            quota,
            cells,
            at,
            outcome: Outcome::Allowed {
                remaining_burst_capacity,
            },
        }
    }

    /// Creates a negative (non-conforming) decision.
    pub(crate) fn denied(quota: Quota, cells: NonZeroU32, at: P, retry_at: P) -> Self {
        Decision {
            quota,
            cells,
            at,
            outcome: Outcome::Denied { retry_at },
        }
    }

    /// Returns `true` if the cells were allowed through the rate limiter.
    #[inline]
    pub fn is_allowed(&self) -> bool {
        matches!(self.outcome, Outcome::Allowed { .. })
    }

    /// Returns the rate limiting [`Quota`] the decision was reached under.
    #[inline]
    pub fn quota(&self) -> Quota {
        self.quota
    }

    /// Returns the number of cells the decision was made for.
    #[inline]
    pub fn cells(&self) -> NonZeroU32 {
        self.cells
    }

    /// Returns the clock reading at which the decision was made.
    ///
    /// This is the exact clock measurement that the rate limiter used to
    /// reach the decision; all other fields of the `Decision` are consistent
    /// with this point in time.
    #[inline]
    pub fn time_of_decision(&self) -> P {
        self.at
    }

    /// Returns the number of cells that could be let through in addition to
    /// this decision, immediately after it was made.
    ///
    /// This returns `Some` only if the decision was positive; negative
    /// decisions (which never consume any capacity) return `None`.
    #[inline]
    pub fn remaining_burst_capacity(&self) -> Option<u32> {
        match self.outcome {
            Outcome::Allowed {
                remaining_burst_capacity,
            } => Some(remaining_burst_capacity),
            Outcome::Denied { .. } => None,
        }
    }

    /// Returns the earliest time at which a decision for the same number of
    /// cells could be positive.
    ///
    /// This returns `Some` only if the decision was negative; positive
    /// decisions return `None`.
    ///
    /// As with [`NotUntil`][crate::NotUntil], this excludes the effect of
    /// other decisions that are made in the meantime.
    #[inline]
    pub fn retry_at(&self) -> Option<P> {
        match self.outcome {
            Outcome::Allowed { .. } => None,
            Outcome::Denied { retry_at } => Some(retry_at),
        }
    }

    /// Returns the minimum amount of time from `from` that must pass before
    /// a decision for the same number of cells could be positive.
    ///
    /// Returns a zero `Duration` if the decision was positive, or if the
    /// earliest retry time lies in the past relative to `from` (which can
    /// happen if the clock has moved backwards since the decision was made).
    #[inline]
    pub fn wait_time_from(&self, from: P) -> Duration {
        match self.outcome {
            Outcome::Allowed { .. } => Duration::from_nanos(0),
            Outcome::Denied { retry_at } => retry_at.duration_since(retry_at.min(from)).into(),
        }
    }
}
