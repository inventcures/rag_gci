package org.inventcures.pallisahayak

import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.LocalViewModelStoreOwner
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.compose.material3.Text

import org.inventcures.pallisahayak.app.ui.AppScaffold
import org.inventcures.pallisahayak.app.ui.HistoryList
import org.inventcures.pallisahayak.app.ui.OfflineBanner
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.ServiceLocator
import org.inventcures.pallisahayak.data.offline.HistoryEntry

/**
 * Wiring for the app shell.
 *
 * A ServiceLocator rather than a DI framework, deliberately. The app has one object
 * graph today, and a container would be configuration written before there is
 * anything to configure. If the graph grows past a dozen nodes, Hilt becomes the
 * right answer.
 */
@Composable
fun AppRoot() {
    // The store owner is the Activity, so the ViewModel survives configuration
    // changes and is cleared when the Activity finishes.
    val storeOwner = LocalViewModelStoreOwner.current
    val viewModel: AskViewModel? = storeOwner?.let { owner ->
        ViewModelProvider(
            owner = owner,
            factory = viewModelFactory {
                initializer {
                    AskViewModel(ServiceLocator.repository, ServiceLocator.currentSession)
                }
            },
        )[AskViewModel::class.java]
    }

    if (viewModel == null) {
        // Only reachable outside an Activity, which in this app means a preview.
        Text("Palli Sahayak is starting")
        return
    }

    val graph = ServiceLocator.offline

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
        settings = { Text("Settings") },
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