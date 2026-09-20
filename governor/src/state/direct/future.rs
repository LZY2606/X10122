use std::num::NonZeroU32;

use super::RateLimiter;
use crate::{
    clock,
    errors::InsufficientCapacity,
    middleware::RateLimitingMiddleware,
    state::{DirectStateStore, NotKeyed},
    Decision, Jitter, NotUntil,
};
use futures_timer::Delay;

#[cfg(feature = "std")]
/// # Direct rate limiters - `async`/`await`
impl<S, C, MW> RateLimiter<NotKeyed, S, C, MW>
where
    S: DirectStateStore,
    C: clock::ReasonablyRealtime,
    MW: RateLimitingMiddleware<C::Instant, NegativeOutcome = NotUntil<C::Instant>>,
{
    /// Asynchronously resolves as soon as the rate limiter allows it.
    ///
    /// When polled, the returned future either resolves immediately (in the case where the rate
    /// limiter allows it), or else triggers an asynchronous delay, after which the rate limiter
    /// is polled again. This means that the future might resolve at some later time (depending
    /// on what other measurements are made on the rate limiter).
    ///
    /// If multiple futures are dispatched against the rate limiter, it is advisable to use
    /// [`until_ready_with_jitter`](#method.until_ready_with_jitter), to avoid thundering herds.
    pub async fn until_ready(&self) -> MW::PositiveOutcome {
        self.until_ready_with_jitter(Jitter::NONE).await
    }

    /// Asynchronously resolves as soon as the rate limiter allows it, with a randomized wait
    /// period.
    ///
    /// When polled, the returned future either resolves immediately (in the case where the rate
    /// limiter allows it), or else triggers an asynchronous delay, after which the rate limiter
    /// is polled again. This means that the future might resolve at some later time (depending
    /// on what other measurements are made on the rate limiter).
    ///
    /// This method allows for a randomized additional delay between polls of the rate limiter,
    /// which can help reduce the likelihood of thundering herd effects if multiple tasks try to
    /// wait on the same rate limiter.
    pub async fn until_ready_with_jitter(&self, jitter: Jitter) -> MW::PositiveOutcome {
        loop {
            match self.check() {
                Ok(x) => {
                    return x;
                }
                Err(negative) => {
                    let delay = Delay::new(jitter + negative.wait_time_from(self.clock.now()));
                    delay.await;
                }
            }
        }
    }

    /// Asynchronously resolves as soon as the rate limiter allows it.
    ///
    /// This is similar to `until_ready` except it waits for an abitrary number
    /// of `n` cells to be available.
    ///
    /// Returns `InsufficientCapacity` if the `n` provided exceeds the maximum
    /// capacity of the rate limiter.
    pub async fn until_n_ready(
        &self,
        n: NonZeroU32,
    ) -> Result<MW::PositiveOutcome, InsufficientCapacity> {
        self.until_n_ready_with_jitter(n, Jitter::NONE).await
    }

    /// Asynchronously resolves as soon as the rate limiter allows it, with a
    /// randomized wait period.
    ///
    /// This is similar to `until_ready_with_jitter` except it waits for an
    /// abitrary number of `n` cells to be available.
    ///
    /// Returns `InsufficientCapacity` if the `n` provided exceeds the maximum
    /// capacity of the rate limiter.
    pub async fn until_n_ready_with_jitter(
        &self,
        n: NonZeroU32,
        jitter: Jitter,
    ) -> Result<MW::PositiveOutcome, InsufficientCapacity> {
        loop {
            match self.check_n(n)? {
                Ok(x) => {
                    return Ok(x);
                }
                Err(negative) => {
                    let delay = Delay::new(jitter + negative.wait_time_from(self.clock.now()));
                    delay.await;
                }
            }
        }
    }
}

#[cfg(feature = "std")]
/// # Direct rate limiters - `async`/`await` with structured decisions
impl<S, C, MW> RateLimiter<NotKeyed, S, C, MW>
where
    S: DirectStateStore,
    C: clock::ReasonablyRealtime,
    MW: RateLimitingMiddleware<C::Instant>,
{
    /// Asynchronously resolves with a structured [`Decision`] as soon as the
    /// rate limiter allows it.
    ///
    /// This behaves like [`until_ready`](#method.until_ready), but resolves
    /// to the [`Decision`] of the final, positive rate-limiting decision:
    /// the same structured evidence that
    /// [`check_with_decision`](struct.RateLimiter.html#method.check_with_decision)
    /// returns.
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "std")]
    /// # fn main () {
    /// use governor::{Quota, RateLimiter};
    /// use nonzero_ext::nonzero;
    ///
    /// let lim = RateLimiter::direct(Quota::per_second(nonzero!(1_u32)));
    /// let decision = futures_executor::block_on(lim.until_ready_with_decision());
    /// assert!(decision.is_allowed());
    /// # }
    /// # #[cfg(not(feature = "std"))]
    /// # fn main() {}
    /// ```
    pub async fn until_ready_with_decision(&self) -> Decision<C::Instant> {
        loop {
            let decision = self.check_with_decision();
            if decision.is_allowed() {
                return decision;
            }
            let delay = Delay::new(decision.wait_time_from(self.clock.now()));
            delay.await;
        }
    }

    /// Asynchronously resolves with a structured [`Decision`] as soon as the
    /// rate limiter allows `n` cells through.
    ///
    /// This behaves like [`until_n_ready`](#method.until_n_ready), but
    /// resolves to the [`Decision`] of the final, positive rate-limiting
    /// decision.
    ///
    /// Returns `InsufficientCapacity` if the `n` provided exceeds the maximum
    /// capacity of the rate limiter.
    pub async fn until_n_ready_with_decision(
        &self,
        n: NonZeroU32,
    ) -> Result<Decision<C::Instant>, InsufficientCapacity> {
        loop {
            let decision = self.check_n_with_decision(n)?;
            if decision.is_allowed() {
                return Ok(decision);
            }
            let delay = Delay::new(decision.wait_time_from(self.clock.now()));
            delay.await;
        }
    }
}

#[cfg(test)]
mod test {
    use assertables::assert_gt;

    use super::*;

    #[test]
    fn insufficient_capacity_impl_coverage() {
        let i = InsufficientCapacity(1);
        assert_eq!(i.0, i.0);
        assert_gt!(format!("{i}").len(), 0);
    }
}
