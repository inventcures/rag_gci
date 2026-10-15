package org.inventcures.pallisahayak.data.release

/**
 * The last release status the server confirmed as usable with participants.
 *
 * A release build that starts with no signal opens on this instead of refusing,
 * so the Cached Answer Bundle can answer offline. It holds only a status that
 * was usable and not drifting; any status the server reports that is not usable
 * clears it, so a release seen unapproved is refused offline too. A store must
 * also forget it when the app is updated, because a status confirmed for one
 * build says nothing about the next.
 */
interface ReleaseStatusCache {
    /** The remembered status, or null when none is held for this build. */
    fun lastApproved(): ReleaseStatus?

    fun remember(status: ReleaseStatus)

    fun forget()
}

/** Held in memory only: the default, and for tests. */
class InMemoryReleaseStatusCache : ReleaseStatusCache {
    @Volatile
    private var status: ReleaseStatus? = null

    override fun lastApproved(): ReleaseStatus? = status

    override fun remember(status: ReleaseStatus) {
        this.status = status
    }

    override fun forget() {
        status = null
    }
}
