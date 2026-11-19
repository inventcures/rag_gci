package org.inventcures.pallisahayak.data.offline

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Whether the device currently has a usable network.
 *
 * Criterion 9: the app recovers cleanly when signal returns, without a restart. That
 * starts with knowing when it came back, so the two reasons the app goes offline have
 * to be visible: no network, and a network that claims to be up but has no internet
 * behind it.
 *
 * The second case is the one that matters in the field. A site with a bar of signal
 * and no working link reports itself connected, the app believes it is online, sends
 * a request, and stalls. Treating "validated" as the question rather than "connected"
 * is what stops the app telling a worker everything is fine while nothing loads.
 *
 * Deliberately no dependency on a connectivity library. The app carries an explicit
 * APK budget for 2G devices, and this is the whole of the functionality needed.
 */
class ConnectivityMonitor(context: Context) {

    private val manager =
        context.applicationContext.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager

    private val _online = MutableStateFlow(currentlyOnline())
    val online: StateFlow<Boolean> = _online.asStateFlow()

    /**
     * Follows the default network only, the one requests actually go out on.
     *
     * It used to watch every network and to report online the moment one appeared.
     * That said "online" before the network was validated, which is the stall this
     * class exists to prevent, and let a second, unvalidated network (a captive
     * Wi-Fi next to working mobile data) flip the app offline. On the default
     * network, onCapabilitiesChanged always follows onAvailable, so that is where
     * validation is read.
     */
    private val callback = object : ConnectivityManager.NetworkCallback() {
        override fun onLost(network: Network) {
            _online.value = currentlyOnline()
        }

        override fun onCapabilitiesChanged(
            network: Network,
            capabilities: NetworkCapabilities,
        ) {
            _online.value = capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        }
    }

    init {
        manager?.registerDefaultNetworkCallback(callback)
    }

    /**
     * True only when a network exists AND is validated.
     *
     * `NET_CAPABILITY_VALIDATED` is the part that matters. A connected network that
     * cannot reach anything reports connected and is what makes an app look frozen.
     */
    fun currentlyOnline(): Boolean {
        val active = manager?.activeNetwork ?: return false
        val capabilities = manager.getNetworkCapabilities(active) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    fun close() {
        runCatching { manager?.unregisterNetworkCallback(callback) }
    }
}