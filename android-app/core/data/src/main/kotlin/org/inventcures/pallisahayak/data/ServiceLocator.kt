package org.inventcures.pallisahayak.data

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import org.inventcures.pallisahayak.data.auth.ApiClients
import org.inventcures.pallisahayak.data.auth.KeystoreTokenStore
import org.inventcures.pallisahayak.data.auth.SessionManager
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.offline.ConnectivityMonitor
import org.inventcures.pallisahayak.data.offline.HistoryRepository
import org.inventcures.pallisahayak.data.offline.OfflineAnswerResolver
import org.inventcures.pallisahayak.data.release.ReleaseGate
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.start.AppStart

/**
 * Builds the object graph.
 *
 * A ServiceLocator rather than a DI framework, deliberately. The app has one
 * object graph today, and a container would be configuration written before there
 * is anything to configure. If the graph grows past a dozen nodes, Hilt becomes
 * the right answer.
 */
object ServiceLocator {

    /**
     * The server, from the build config (`-PpalliBaseUrl`, see core/data/build.gradle.kts).
     *
     * A release build refuses to configure with a non-https or `.invalid` URL,
     * so a build cannot ship pointing nowhere.
     */
    val baseUrl: String = BuildConfig.BASE_URL

    /** The study site this build is for (`-PpalliSiteId`), sent at Registration. */
    val siteId: String = BuildConfig.SITE_ID

    @Volatile
    private var repositoryInstance: PalliSahayakRepository? = null

    @Volatile
    private var offlineGraph: OfflineGraph? = null

    @Volatile
    private var clients: ApiClients? = null

    @Volatile
    private var startInstance: AppStart? = null

    /**
     * Everything the offline screens need, built once alongside the repository.
     *
     * Grouped rather than exposed as four more properties because they share one
     * database handle and are meaningless apart from each other.
     */
    val offline: OfflineGraph
        get() = offlineGraph ?: notInitialised()

    val repository: PalliSahayakRepository
        get() = repositoryInstance ?: notInitialised()

    /** Registration, PIN sign-in and the token the API calls carry. */
    val sessions: SessionManager
        get() = clients?.sessions ?: notInitialised()

    /** The release gate and the first screen. */
    val start: AppStart
        get() = startInstance ?: notInitialised()

    @Synchronized
    fun initialise(context: Context) {
        if (repositoryInstance != null) return

        // BASIC logs the request line and status, never headers, so the bearer
        // token does not reach logcat.
        val base = OkHttpClient.Builder()
            .addInterceptor(
                HttpLoggingInterceptor().apply {
                    level = HttpLoggingInterceptor.Level.BASIC
                },
            )
            .build()

        val api = ApiClients(baseUrl = baseUrl, tokens = KeystoreTokenStore(context), base = base)
        clients = api

        val database = PalliDatabase.get(context)

        repositoryInstance = PalliSahayakRepository(
            api = api.api,
            interactions = database.interactions(),
        )

        // A debug build always opens (ReleaseGate explains why); a release build
        // opens only on a release the server confirms as approved and activated.
        startInstance = AppStart(
            api = api.authApi,
            gate = ReleaseGate(developmentBuild = BuildConfig.DEBUG),
            sessions = api.sessions,
        )

        offlineGraph = OfflineGraph(
            history = HistoryRepository(database.interactions()),
            bundle = OfflineAnswerResolver(database.cachedAnswers()),
            connectivity = ConnectivityMonitor(context),
        )
    }

    private fun notInitialised(): Nothing = error(
        "ServiceLocator.initialise(context) has not been called. " +
            "MainActivity should do this in onCreate before setContent.",
    )
}

/**
 * The offline object graph.
 *
 * A small named holder so the screens take one dependency rather than three, and so
 * a test can build the same shape without an Activity.
 */
data class OfflineGraph(
    val history: HistoryRepository,
    val bundle: OfflineAnswerResolver,
    val connectivity: ConnectivityMonitor,
)
