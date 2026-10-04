package org.inventcures.pallisahayak.data.release

/**
 * What the server says about this build.
 *
 * Mirrors `StudyRelease.client_view()` in `study_release.py`. Duplicated rather than
 * generated because the client cannot import server code, and pinned by
 * `ReleaseGateTest` against the recorded shape.
 */
data class ReleaseStatus(
    val releaseId: String,
    val status: String,
    val activated: Boolean,
    val approved: Boolean,
    val usableWithParticipants: Boolean,
    val drifting: Boolean,
    val drift: List<String> = emptyList(),
) {
    companion object {
        /**
         * The value used when the server cannot be reached.
         *
         * Denies. A build that cannot confirm its own approval must not be used with
         * participants, and the failure this protects against is not theoretical: an
         * unapproved build answering a real patient's question is exactly what the
         * study's precondition is there to prevent.
         */
        fun unknown(): ReleaseStatus = ReleaseStatus(
            releaseId = "unknown",
            status = "unknown",
            activated = false,
            approved = false,
            usableWithParticipants = false,
            drifting = false,
        )
    }
}

/**
 * Why the app will not open, or that it will.
 *
 * A refusal the worker cannot act on is a support call, so each state carries the
 * reason and, where there is one, what to do instead. Criterion 6 says the app says
 * why, and "it refuses" alone does not satisfy that.
 */
sealed interface GateDecision {

    /** Safe to use with participants. */
    object Open : GateDecision

    /**
     * Refused, and the reason is one the owner can fix.
     */
    data class Blocked(
        val headline: String,
        val detail: String,
        /** True when the cause is a record that is unsigned rather than a bad build. */
        val awaitingApproval: Boolean = false,
    ) : GateDecision
}

/**
 * The gate the app opens behind.
 *
 * Criterion 6: the app refuses to start against a release that is not approved, and
 * says why. Criterion 7: drift is visible to the study team before any participant
 * uses the build.
 *
 * The distinction that matters here is approved versus activated. A release can be
 * signed off by both leads and still not be open to participants, and treating those
 * as one state would either let an unactivated release into the field or block a
 * perfectly good build for no reason.
 */
class ReleaseGate(
    /** True when this build is a development build rather than a study build. */
    private val developmentBuild: Boolean = true,
) {

    /**
     * Decide whether the app may open.
     *
     * A development build always opens. This is not a loophole: the gate exists so no
     * participant is ever given an unapproved system, and nobody is a participant on
     * a developer's own machine.
     */
    fun evaluate(status: ReleaseStatus): GateDecision {
        if (developmentBuild) return GateDecision.Open

        // Drift is checked before approval, not after. An approved build that no
        // longer matches its approved record is the dangerous one: it carries a
        // signature for a system that is not the one running. Criterion 7 wants that
        // visible before any participant uses it, so drift has to win over a signed
        // release, or the gate passes exactly the build it exists to catch.
        if (status.drifting) {
            return GateDecision.Blocked(
                headline = "This build has changed since it was approved",
                detail = "It no longer matches the approved record. The study team " +
                    "needs to look at this before anyone uses it.",
            )
        }

        if (status.usableWithParticipants) return GateDecision.Open

        if (!status.approved) {
            return GateDecision.Blocked(
                headline = "This version is not approved yet",
                detail = "It has not been signed off by the study team. Nothing is " +
                    "lost, and it will open once it is approved.",
                awaitingApproval = true,
            )
        }

        if (!status.activated) {
            return GateDecision.Blocked(
                headline = "This version is not open yet",
                detail = "It is approved but the study has not started using it yet.",
                awaitingApproval = true,
            )
        }

        return GateDecision.Blocked(
            headline = "This version cannot be used",
            detail = "The study team has to resolve this before it is used.",
        )
    }
}