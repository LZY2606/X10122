//! Direct rate limiters (those that can only hold one state).
//!
//! Rate limiters based on these types are constructed with
//! [the `RateLimiter` constructors](../struct.RateLimiter.html#direct-in-memory-rate-limiters---constructors)

use core::num::NonZeroU32;

use crate::{
    clock,
    errors::InsufficientCapacity,
    middleware::{NoOpMiddleware, RateLimitingMiddleware},
    state::InMemoryState,
    Decision, Quota,
};

/// The "this state store does not use keys" key type.
///
/// It's possible to use this to create a "direct" rate limiter. It explicitly does not implement
/// [`Hash`][std::hash::Hash] so that it is possible to tell apart from "hashable" key types.
#[derive(PartialEq, Debug, Eq)]
pub enum NotKeyed {
    /// The value given to state stores' methods.
    NonKey,
}

/// A trait for state stores that only keep one rate limiting state.
///
/// This is blanket-implemented by all [`StateStore`]s with [`NotKeyed`] key associated types.
pub trait DirectStateStore: StateStore<Key = NotKeyed> {}

impl<T> DirectStateStore for T where T: StateStore<Key = NotKeyed> {}

/// # Direct in-memory rate limiters - Constructors
///
/// Here we construct an in-memory rate limiter that makes direct (un-keyed)
/// rate-limiting decisions. Direct rate limiters can be used to
/// e.g. regulate the transmission of packets on a single connection,
/// or to ensure that an API client stays within a service's rate
/// limit.
#[cfg(feature = "std")]
impl RateLimiter<NotKeyed, InMemoryState, clock::DefaultClock, NoOpMiddleware> {
    /// Constructs a new in-memory direct rate limiter for a quota with the default real-time clock.
    pub fn direct(
        quota: Quota,
    ) -> RateLimiter<NotKeyed, InMemoryState, clock::DefaultClock, NoOpMiddleware> {
        let clock = clock::DefaultClock::default();
        Self::direct_with_clock(quota, clock)
    }
}

impl<C> RateLimiter<NotKeyed, InMemoryState, C, NoOpMiddleware<C::Instant>>
where
    C: clock::Clock,
{
    /// Constructs a new direct rate limiter for a quota with a custom clock.
    pub fn direct_with_clock(quota: Quota, clock: C) -> Self {
        let state: InMemoryState = Default::default();
        RateLimiter::new(quota, state, clock)
    }
}

/// # Direct rate limiters - Manually checking cells
impl<S, C, MW> RateLimiter<NotKeyed, S, C, MW>
where
    S: DirectStateStore,
    C: clock::Clock,
    MW: RateLimitingMiddleware<C::Instant>,
{
    /// Allow a single cell through the rate limiter.
    ///
    /// If the rate limit is reached, `check` returns information about the earliest
    /// time that a cell might be allowed through again.
    pub fn check(&self) -> Result<MW::PositiveOutcome, MW::NegativeOutcome> {
        self.gcra.test_and_update::<NotKeyed, C::Instant, S, MW>(
            self.start,
            &NotKeyed::NonKey,
            &self.state,
            self.clock.now(),
        )
    }

    /// Allow *only all* `n` cells through the rate limiter.
    ///
    /// This method can succeed in only one way and fail in two ways:
    /// * Success: If all `n` cells can be accommodated, it returns `Ok(())`.
    /// * Failure (but ok): Not all cells can make it through at the current time.
    ///   The result is `Err(NegativeMultiDecision::BatchNonConforming(NotUntil))`, which can
    ///   be interrogated about when the batch might next conform.
    /// * Failure (the batch can never go through): The rate limit quota's burst size is too low
    ///   for the given number of cells to ever be allowed through.
    ///
    /// ### Performance
    /// This method diverges a little from the GCRA algorithm, using
    /// multiplication to determine the next theoretical arrival time, and so
    /// is not as fast as checking a single cell.
    pub fn check_n(
        &self,
        n: NonZeroU32,
    ) -> Result<Result<MW::PositiveOutcome, MW::NegativeOutcome>, InsufficientCapacity> {
        self.gcra
            .test_n_all_and_update::<NotKeyed, C::Instant, S, MW>(
                self.start,
                &NotKeyed::NonKey,
                n,
                &self.state,
                self.clock.now(),
            )
    }

    /// Allow a single cell through the rate limiter and return a structured
    /// [`Decision`] describing the outcome.
    ///
    /// Unlike [`check`](#method.check), which only reports whether the cell
    /// was allowed through, the returned [`Decision`] captures the evidence
    /// of the verdict as observed at the moment of the decision: the time
    /// the decision was made, the quota it was reached under, and either the
    /// remaining burst capacity (if the cell was allowed) or the earliest
    /// time a retry could be allowed (if it was not).
    ///
    /// All of this information is derived from the *same* state transition,
    /// so concurrent decisions made by other threads can not skew the values
    /// relative to each other. A negative decision never consumes any
    /// capacity.
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "std")]
    /// # fn main () {
    /// use governor::{Quota, RateLimiter};
    /// use nonzero_ext::nonzero;
    ///
    /// let lim = RateLimiter::direct(Quota::per_second(nonzero!(1_u32)));
    /// let decision = lim.check_with_decision();
    /// assert!(decision.is_allowed());
    /// assert_eq!(decision.remaining_burst_capacity(), Some(0));
    ///
    /// let decision = lim.check_with_decision();
    /// assert!(!decision.is_allowed());
    /// assert!(decision.retry_at().is_some());
    /// # }
    /// # #[cfg(not(feature = "std"))]
    /// # fn main() {}
    /// ```
    pub fn check_with_decision(&self) -> Decision<C::Instant> {
        self.gcra.test_and_update_decision(
            self.start,
            &NotKeyed::NonKey,
            &self.state,
            self.clock.now(),
        )
    }

    /// Allow *only all* `n` cells through the rate limiter and return a
    /// structured [`Decision`] describing the outcome.
    ///
    /// This behaves exactly like [`check_n`](#method.check_n), but reports
    /// the outcome as a [`Decision`]: If all `n` cells can be accommodated,
    /// the decision is positive and reports the remaining burst capacity;
    /// otherwise it is negative and reports the earliest time the batch
    /// could conform. As with `check_n`, if the batch can never conform
    /// because it exceeds the quota's burst size, this returns
    /// [`InsufficientCapacity`].
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "std")]
    /// # fn main () {
    /// use governor::{Quota, RateLimiter};
    /// use nonzero_ext::nonzero;
    ///
    /// let lim = RateLimiter::direct(
    ///     Quota::per_second(nonzero!(1_u32)).allow_burst(nonzero!(4_u32)),
    /// );
    /// let decision = lim.check_n_with_decision(nonzero!(3_u32)).unwrap();
    /// assert!(decision.is_allowed());
    /// assert_eq!(decision.cells(), nonzero!(3_u32));
    /// assert_eq!(decision.remaining_burst_capacity(), Some(1));
    /// # }
    /// # #[cfg(not(feature = "std"))]
    /// # fn main() {}
    /// ```
    pub fn check_n_with_decision(
        &self,
        n: NonZeroU32,
    ) -> Result<Decision<C::Instant>, InsufficientCapacity> {
        self.gcra.test_n_all_and_update_decision(
            self.start,
            &NotKeyed::NonKey,
            n,
            &self.state,
            self.clock.now(),
        )
    }
}

#[cfg(feature = "std")]
mod future;

#[cfg(feature = "std")]
mod sinks;
#[cfg(feature = "std")]
pub use sinks::*;

#[cfg(feature = "std")]
mod streams;

use crate::state::{RateLimiter, StateStore};
#[cfg(feature = "std")]
pub use streams::*;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn not_keyed_impls_coverage() {
        assert_eq!(NotKeyed::NonKey, NotKeyed::NonKey);
    }
}
