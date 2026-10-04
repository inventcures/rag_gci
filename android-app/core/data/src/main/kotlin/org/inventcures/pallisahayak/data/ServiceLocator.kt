package org.inventcures.pallisahayak.data

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.data.local.PalliDatabase
import org.inventcures.pallisahayak.data.offline.ConnectivityMonitor
import org.inventcures.pallisahayak.data.offline.HistoryRepository
import org.inventcures.pallisahayak.data.offline.OfflineAnswerResolver
import org.inventcures.pallisahayak.data.repository.PalliSahayakRepository
import org.inventcures.pallisahayak.data.repository.SessionContext
import retrofit2.Retrofit
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import retrofit2.converter.moshi.MoshiConverterFactory

/**
 * Builds the object graph.
 *
 * A ServiceLocator rather than a DI framework, deliberately. The app has one
 * object graph today, and a container would be configuration written before there
 * is anything to configure. If the graph grows past a dozen nodes, Hilt becomes
 * the right answer.
 */
object ServiceLocator {

    /** Base URL. Overridable per build so a study deployment can point elsewhere. */
    private const val BASE_URL = "https://palli-sahayak.invalid/api/mobile/v1/"

    /**
     * Placeholder until registration exists.
     *
     * A participant's id, role and release must come from registration rather than
     * a constant. Until then `releaseApproved` is false, so anything recorded
     * against this session is correctly marked as not from an approved release.
     * Ticket 06 replaces this.
     */
    val currentSession = SessionContext(
        participantId = "placeholder",
        careRole = "ASHA Worker",
        siteId = "unassigned",
        language = "hi-IN",
        releaseId = "unassigned-dev",
        releaseApproved = false,
    )

    @Volatile
    private var repositoryInstance: PalliSahayakRepository? = null

    @Volatile
    private var offlineGraph: OfflineGraph? = null

    /**
     * Everything the offline screens need, built once alongside the repository.
     *
     * Grouped rather than exposed as four more properties because they share one
     * database handle and are meaningless apart from each other.
     */
    val offline: OfflineGraph
        get() = offlineGraph ?: error(
            "ServiceLocator.initialise(context) has not been called. " +
                "MainActivity should do this in onCreate before setContent.",
        )

    val repository: PalliSahayakRepository
        get() = repositoryInstance ?: error(
            "ServiceLocator.initialise(context) has not been called. " +
                "MainActivity should do this in onCreate before setContent.",
        )

    @Synchronized
    fun initialise(context: Context, api: PalliSahayakApi? = null) {
        if (repositoryInstance != null) return

        val client = OkHttpClient.Builder()
            .addInterceptor(
                HttpLoggingInterceptor().apply {
                    level = HttpLoggingInterceptor.Level.BASIC
                },
            )
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(
                // KotlinJsonAdapterFactory is required. Without it Moshi's
                // reflective adapter cannot read Kotlin default parameter values,
                // so the generated data classes fail to deserialise and every
                // request falls into the offline branch with no error anywhere.
                MoshiConverterFactory.create(
                    Moshi.Builder().add(KotlinJsonAdapterFactory()).build(),
                ),
            )
            .build()

        val database = PalliDatabase.get(context)

        repositoryInstance = PalliSahayakRepository(
            api = api ?: retrofit.create(PalliSahayakApi::class.java),
            interactions = database.interactions(),
        )

        offlineGraph = OfflineGraph(
            history = HistoryRepository(database.interactions()),
            bundle = OfflineAnswerResolver(database.cachedAnswers()),
            connectivity = ConnectivityMonitor(context),
        )
    }
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
