package org.inventcures.pallisahayak

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import org.inventcures.pallisahayak.data.ServiceLocator

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Built before setContent, because the composition reads the repository.
        ServiceLocator.initialise(applicationContext)
        setContent { AppRoot() }
    }
}
