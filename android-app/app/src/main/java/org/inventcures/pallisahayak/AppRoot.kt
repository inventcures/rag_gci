package org.inventcures.pallisahayak

import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.viewmodel.compose.LocalViewModelStoreOwner
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.compose.material3.Text
import androidx.compose.ui.res.stringResource

import org.inventcures.pallisahayak.app.ui.AppScaffold
import org.inventcures.pallisahayak.app.ui.HistoryList
import org.inventcures.pallisahayak.app.ui.LanguagePicker
import org.inventcures.pallisahayak.app.ui.OfflineBanner
import org.inventcures.pallisahayak.app.ui.RegistrationScreen
import org.inventcures.pallisahayak.app.ui.ReleaseRefusalScreen
import org.inventcures.pallisahayak.app.ui.SettingsScreen
import org.inventcures.pallisahayak.app.ui.SignInScreen
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.ServiceLocator
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.model.Language
import org.inventcures.pallisahayak.data.offline.HistoryEntry
import org.inventcures.pallisahayak.data.release.ReleaseStatus
import org.inventcures.pallisahayak.data.repository.SessionContext
import org.inventcures.pallisahayak.data.settings.ActiveSettings
import org.inventcures.pallisahayak.data.start.StartScreen

/**
 * Wiring for the app shell.
 *
 * Settings switch the Active Role and the language ([ActiveSettings]); the Ask
 * ViewModel reads them for every question, so the next interaction is recorded
 * under them, and a language switch redraws the screens in that language.
 *
 * The release gate runs first, once per launch: a release the server does not
 * confirm opens on the refusal screen and nothing else (criterion 6). Then the
 * sign-in state picks the screen: registration (language, then Care Role and
 * PIN), PIN sign-in once a session has ended, or the app itself, whose every
 * interaction is recorded against the signed-in participant.
 *
 * A ServiceLocator rather than a DI framework, deliberately. The app has one object
 * graph today, and a container would be configuration written before there is
 * anything to configure. If the graph grows past a dozen nodes, Hilt becomes the
 * right answer.
 */
@Composable
fun AppRoot() {
    // Every screen below is shown in the chosen language, and redrawn in place
    // when settings switch it.
    val language: Language? by ServiceLocator.settings.language.collectAsState()
    InLanguage(language) { AppScreens(language) }
}

@Composable
private fun AppScreens(language: Language?) {
    // The store owner is the Activity, so the ViewModel survives configuration
    // changes and is cleared when the Activity finishes.
    val storeOwner = LocalViewModelStoreOwner.current
    if (storeOwner == null) {
        // Only reachable outside an Activity, which in this app means a preview.
        Text(stringResource(R.string.starting_up))
        return
    }

    val start = ServiceLocator.start
    val sessions = ServiceLocator.sessions
    val settings = ServiceLocator.settings
    var gateChecks by rememberSaveable { mutableIntStateOf(0) }
    val release: ReleaseStatus? by produceState<ReleaseStatus?>(null, gateChecks) {
        value = start.release()
    }
    val auth by sessions.state.collectAsState()

    val checked = release
    if (checked == null) {
        Text(stringResource(R.string.starting_up))
        return
    }

    when (val screen = start.screenFor(checked, auth)) {
        is StartScreen.Refused -> ReleaseRefusalScreen(screen.decision, onRetry = { gateChecks++ })
        StartScreen.Register -> {
            val chosen = language
            if (chosen == null) {
                // Chosen once, then the screens are shown in it from registration on.
                LanguagePicker(selected = null, onSelect = settings::switchLanguage)
            } else {
                RegistrationScreen(onRegister = { role, pin ->
                    sessions.register(role, pin, chosen, ServiceLocator.siteId)
                })
            }
        }
        StartScreen.SignIn -> SignInScreen(onSignIn = { pin -> sessions.signIn(pin) })
        is StartScreen.Ready -> SignedInApp(storeOwner, screen.session, language)
    }
}

/** The app itself, for one signed-in participant. */
@Composable
private fun SignedInApp(storeOwner: ViewModelStoreOwner, session: SessionContext, language: Language?) {
    // Keyed by participant, so a different person signing in on this phone never
    // inherits the previous person's screen state or session.
    val viewModel = ViewModelProvider(
        owner = storeOwner,
        factory = viewModelFactory {
            initializer {
                AskViewModel(ServiceLocator.repository, session, settings = ServiceLocator.settings)
            }
        },
    )["ask:${session.participantId}", AskViewModel::class.java] // not user-facing: a ViewModel store key

    val graph = ServiceLocator.offline
    val settings = ServiceLocator.settings
    val switchedRole: CareRole? by settings.role.collectAsState()

    // Criterion 9 is only reachable because this subscribes to the monitor rather
    // than sampling once at startup. Sampling would leave the app believing it is
    // offline after signal returns until something else forced a redraw.
    val online: Boolean by graph.connectivity.online.collectAsStateSafe(initial = false)
    val history: List<HistoryEntry> by graph.history.observe().collectAsStateSafe(emptyList())

    AppScaffold(
        offlineBanner = {
            if (!online) {
                OfflineBanner(cachedCount = history.size)
            }
        },
        ask = { AskScreen(viewModel, offline = !online) },
        history = { HistoryList(entries = history) },
        settings = {
            SettingsScreen(
                activeRole = switchedRole ?: CareRole.fromWire(session.careRole),
                onRoleSelect = settings::switchRole,
                language = language ?: Language.fromTag(session.language),
                onLanguageSelect = settings::switchLanguage,
            )
        },
    )
}

/**
 * Collect a flow without pulling in a lifecycle-aware helper for one call site.
 *
 * Returns [initial] until the first emission, so the shell never renders an empty
 * list and then snaps to content.
 */
@Composable
private fun <T> kotlinx.coroutines.flow.Flow<T>.collectAsStateSafe(
    initial: T,
): State<T> = produceState(initial, this) { collect { value = it } }