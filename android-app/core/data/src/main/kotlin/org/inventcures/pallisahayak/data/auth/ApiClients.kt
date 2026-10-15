package org.inventcures.pallisahayak.data.auth

import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.Dispatcher
import okhttp3.OkHttpClient
import org.inventcures.pallisahayak.api.generated.PalliSahayakApi
import org.inventcures.pallisahayak.data.sync.wire.SyncApi
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.Executor

/**
 * The two API clients the app talks to the server with, and the session between them.
 *
 * [api] signs every call with the access token and refreshes it on a 401. It is
 * what the repository uses. [authApi] is unsigned: registration, PIN sign-in,
 * refresh and the release gate go through it, because they are what produce a
 * token or are read before there is one (report 08; `auth/policy.py`).
 *
 * The auth client has its own dispatcher. A refresh runs while the refused call
 * still holds a slot on the main client's dispatcher, and sharing one would let
 * a burst of refused calls starve the refresh that would release them.
 */
class ApiClients(
    baseUrl: String,
    tokens: TokenStore,
    base: OkHttpClient = OkHttpClient(),
) {
    private val authClient: OkHttpClient = base.newBuilder()
        .dispatcher(Dispatcher())
        .build()

    /** The unsigned client: register, login, refresh and release status. */
    val authApi: PalliSahayakApi = retrofit(baseUrl, authClient).create(PalliSahayakApi::class.java)

    val sessions: SessionManager = SessionManager(authApi, tokens)

    /** The signed client, for the voice socket as well as [api]. */
    val client: OkHttpClient = base.newBuilder()
        .addInterceptor(BearerTokenInterceptor(sessions))
        .authenticator(TokenRefreshAuthenticator(sessions))
        .build()

    private val signed: Retrofit = retrofit(baseUrl, client)

    /** The signed API, which the repository asks questions through. */
    val api: PalliSahayakApi = signed.create(PalliSahayakApi::class.java)

    /** The signed sync push, which is not a generated operation. */
    val sync: SyncApi = signed.create(SyncApi::class.java)

    private companion object {
        /**
         * Delivers results on OkHttp's thread rather than the main thread.
         *
         * A suspend call resumes on its own coroutine dispatcher either way, so the
         * hop to the main thread bought nothing. It did make a refresh, which waits
         * for its result on an OkHttp thread, depend on the main thread being free.
         */
        val sameThread = Executor { it.run() }

        fun retrofit(baseUrl: String, client: OkHttpClient): Retrofit = Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(client)
            .callbackExecutor(sameThread)
            .addConverterFactory(
                // KotlinJsonAdapterFactory is required. Without it Moshi's
                // reflective adapter cannot read Kotlin default parameter values,
                // so the generated data classes fail to deserialise and every
                // request falls into the offline branch with no error anywhere.
                MoshiConverterFactory.create(Moshi.Builder().add(KotlinJsonAdapterFactory()).build()),
            )
            .build()
    }
}
