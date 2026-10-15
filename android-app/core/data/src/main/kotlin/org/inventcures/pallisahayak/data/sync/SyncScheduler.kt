package org.inventcures.pallisahayak.data.sync

import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch

/**
 * The background job that empties the sync queue.
 *
 * Whenever the phone has a validated connection it pushes at once, then again
 * every [retryEveryMillis] while it stays online, so an item the server could
 * not save this time ("save_failed", "over_limit") goes on a later push. Losing
 * the connection cancels the wait; regaining it pushes straight away.
 *
 * It runs in the app's process scope rather than WorkManager. That dependency is
 * not in the build and the APK carries a 2G size budget; the trade is that a push
 * waits for the app to be opened, which is when offline answers are made anyway.
 *
 * [onOnline] runs first on each reconnection, for work that also needs signal
 * (downloading the Cached Answer Bundle).
 */
class SyncScheduler(
    private val queue: SyncQueue,
    private val online: Flow<Boolean>,
    private val scope: CoroutineScope,
    private val retryEveryMillis: Long = DEFAULT_RETRY_MILLIS,
    private val onOnline: suspend () -> Unit = {},
) {

    /** Start watching the connection. The job lives as long as [scope]. */
    fun start(): Job = scope.launch {
        online.distinctUntilChanged().collectLatest { up ->
            if (!up) return@collectLatest
            runCatchingLogged("reconnection work") { onOnline() }
            while (true) {
                runCatchingLogged("sync push") { queue.push() }
                delay(retryEveryMillis)
            }
        }
    }

    /**
     * A failure here must not end the job: the next connection should still sync.
     * Cancellation is rethrown, because that is how going offline stops the loop.
     */
    private suspend fun runCatchingLogged(what: String, block: suspend () -> Unit) {
        try {
            block()
        } catch (error: CancellationException) {
            throw error
        } catch (error: RuntimeException) {
            Log.w(TAG, "$what failed; will try again", error)
        }
    }

    private companion object {
        const val TAG = "PalliSync"
        const val DEFAULT_RETRY_MILLIS = 15 * 60 * 1000L
    }
}
