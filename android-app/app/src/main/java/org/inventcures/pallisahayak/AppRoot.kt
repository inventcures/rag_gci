package org.inventcures.pallisahayak

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.LocalViewModelStoreOwner
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.data.ServiceLocator

/**
 * Wiring for the single screen.
 *
 * A ServiceLocator rather than a DI framework, deliberately. The app has one
 * object graph today, and a container would be configuration written before there
 * is anything to configure. If the graph grows past a dozen nodes, Hilt becomes
 * the right answer.
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
    AskScreen(viewModel)
}
