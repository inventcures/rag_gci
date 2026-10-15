package org.inventcures.pallisahayak.data.release

import android.content.Context
import android.content.SharedPreferences

/**
 * The last usable release status, kept across restarts in SharedPreferences.
 *
 * Tied to [buildId], which changes whenever the app is installed or updated: a
 * status the server confirmed for one build is not read back for another, so a
 * new build must reach the server once before it opens offline.
 */
class SharedPreferencesReleaseCache(
    context: Context,
    private val buildId: String,
) : ReleaseStatusCache {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    override fun lastApproved(): ReleaseStatus? {
        if (prefs.getString(KEY_BUILD, null) != buildId) return null
        val releaseId = prefs.getString(KEY_RELEASE, null) ?: return null
        return ReleaseStatus(
            releaseId = releaseId,
            status = prefs.getString(KEY_STATUS, null) ?: return null,
            activated = prefs.getBoolean(KEY_ACTIVATED, false),
            approved = prefs.getBoolean(KEY_APPROVED, false),
            usableWithParticipants = prefs.getBoolean(KEY_USABLE, false),
            drifting = false,
        )
    }

    override fun remember(status: ReleaseStatus) {
        prefs.edit()
            .putString(KEY_BUILD, buildId)
            .putString(KEY_RELEASE, status.releaseId)
            .putString(KEY_STATUS, status.status)
            .putBoolean(KEY_ACTIVATED, status.activated)
            .putBoolean(KEY_APPROVED, status.approved)
            .putBoolean(KEY_USABLE, status.usableWithParticipants)
            .apply()
    }

    override fun forget() {
        prefs.edit().clear().apply()
    }

    private companion object {
        const val NAME = "palli-sahayak-release"
        const val KEY_BUILD = "build_id"
        const val KEY_RELEASE = "release_id"
        const val KEY_STATUS = "status"
        const val KEY_ACTIVATED = "activated"
        const val KEY_APPROVED = "approved"
        const val KEY_USABLE = "usable_with_participants"
    }
}
