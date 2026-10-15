package org.inventcures.pallisahayak.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import org.inventcures.pallisahayak.R
import org.inventcures.pallisahayak.data.auth.AuthResult
import org.inventcures.pallisahayak.data.model.CareRole
import org.inventcures.pallisahayak.data.release.BlockReason
import org.inventcures.pallisahayak.data.release.GateDecision

/** A PIN the server accepts: 4 to 6 digits (`DeviceRegistrationRequest.pin`). */
internal fun isValidPin(pin: String): Boolean = pin.length in 4..6 && pin.all { it.isDigit() }

/**
 * The release gate's refusal, shown instead of the app (criterion 6).
 *
 * Nothing else is reachable from here: no registration, no questions. It says
 * why, and offers to check again, since the usual cause is a release that has
 * not been activated yet or a phone that had no signal at start.
 */
@Composable
fun ReleaseRefusalScreen(decision: GateDecision.Blocked, onRetry: () -> Unit) {
    val (title, detail) = when (decision.reason) {
        BlockReason.DRIFTING -> R.string.release_refused_drifting_title to R.string.release_refused_drifting_detail
        BlockReason.NOT_APPROVED -> R.string.release_refused_not_approved_title to R.string.release_refused_not_approved_detail
        BlockReason.NOT_ACTIVATED -> R.string.release_refused_not_activated_title to R.string.release_refused_not_activated_detail
        BlockReason.UNUSABLE -> R.string.release_refused_unusable_title to R.string.release_refused_unusable_detail
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = stringResource(title),
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Text(text = stringResource(detail), style = MaterialTheme.typography.bodyLarge)
        Button(
            onClick = onRetry,
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
        ) {
            Text(stringResource(R.string.release_retry), style = MaterialTheme.typography.titleMedium)
        }
    }
}

/**
 * Registration: the Care Role and a PIN, typed twice.
 *
 * Typed twice because a mistyped PIN nobody knows is a participant locked out
 * at the next sign-in. No name and no phone number are asked for.
 */
@Composable
fun RegistrationScreen(onRegister: suspend (CareRole, String) -> AuthResult) {
    var role by remember { mutableStateOf(CareRole.ASHA_WORKER) }
    var pin by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf("") }
    var working by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<Int?>(null) }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = stringResource(R.string.register_title),
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Text(text = stringResource(R.string.register_explainer), style = MaterialTheme.typography.bodyMedium)
        CareRole.ALL.forEach { option ->
            val label = stringResource(option.labelRes)
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 56.dp)
                    .clickable { role = option }
                    .semantics { contentDescription = label },
            ) {
                RadioButton(selected = option == role, onClick = { role = option })
                Icon(
                    imageVector = RoleIcons.forRole(option),
                    contentDescription = null,
                    tint = option.tint,
                    modifier = Modifier.size(40.dp),
                )
                Text(
                    text = label,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(start = 12.dp),
                )
            }
        }
        PinField(value = pin, label = R.string.pin_label, onChange = { pin = it })
        PinField(value = confirm, label = R.string.pin_confirm_label, onChange = { confirm = it })
        error?.let { Text(stringResource(it), color = MaterialTheme.colorScheme.error) }
        Button(
            enabled = !working,
            onClick = {
                error = when {
                    !isValidPin(pin) -> R.string.pin_invalid
                    pin != confirm -> R.string.pin_mismatch
                    else -> null
                }
                if (error == null) {
                    working = true
                    scope.launch {
                        error = messageFor(onRegister(role, pin), R.string.register_refused)
                        working = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
        ) {
            Text(stringResource(R.string.register_button), style = MaterialTheme.typography.titleMedium)
        }
    }
}

/** PIN sign-in, for a registered phone whose session has ended. */
@Composable
fun SignInScreen(onSignIn: suspend (String) -> AuthResult) {
    var pin by remember { mutableStateOf("") }
    var working by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<Int?>(null) }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = stringResource(R.string.sign_in_title),
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Text(text = stringResource(R.string.sign_in_explainer), style = MaterialTheme.typography.bodyMedium)
        PinField(value = pin, label = R.string.pin_label, onChange = { pin = it })
        error?.let { Text(stringResource(it), color = MaterialTheme.colorScheme.error) }
        Button(
            enabled = !working,
            onClick = {
                if (!isValidPin(pin)) {
                    error = R.string.pin_invalid
                } else {
                    working = true
                    scope.launch {
                        error = messageFor(onSignIn(pin), R.string.sign_in_refused)
                        if (error != null) pin = ""
                        working = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
        ) {
            Text(stringResource(R.string.sign_in_button), style = MaterialTheme.typography.titleMedium)
        }
    }
}

@Composable
private fun PinField(value: String, label: Int, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = { typed -> onChange(typed.filter { it.isDigit() }.take(6)) },
        label = { Text(stringResource(label)) },
        singleLine = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
        modifier = Modifier.fillMaxWidth(),
    )
}

/** The message to show for a failed attempt, or null when it worked. */
private fun messageFor(result: AuthResult, refused: Int): Int? = when (result) {
    AuthResult.SignedIn -> null
    AuthResult.Refused -> refused
    AuthResult.Unreachable -> R.string.server_unreachable
}
