package org.inventcures.pallisahayak

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.inventcures.pallisahayak.data.AskUiState
import org.inventcures.pallisahayak.data.AskViewModel
import org.inventcures.pallisahayak.safety.AnswerKind
import org.inventcures.pallisahayak.safety.EmergencySeverity

/**
 * The ask screen.
 *
 * Strings are English only for now. The other ten languages are a data change
 * once a native speaker has reviewed them, and shipping machine translations to a
 * patient would be worse than shipping none.
 */
@Composable
fun AskScreen(viewModel: AskViewModel) {
    val state by viewModel.state.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        QuestionInput(
            draft = state.draft,
            enabled = !state.isAsking,
            onDraftChanged = viewModel::onDraftChanged,
            onSubmit = viewModel::submit,
        )

        if (state.emergency.overridesDoseBoundary) {
            EmergencyBanner(state.emergency)
        }

        state.errorMessage?.let {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFFFF4E5),
                ),
            ) {
                Text(
                    text = it,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        AnswerCard(state)

        if (state.history.isNotEmpty()) {
            Text(
                text = "Earlier",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
            )
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.history) { (question, answer) ->
                    HistoryRow(question, answer)
                }
            }
        }
    }
}

@Composable
private fun QuestionInput(
    draft: String,
    enabled: Boolean,
    onDraftChanged: (String) -> Unit,
    onSubmit: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = draft,
            onValueChange = onDraftChanged,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Ask a question") },
            minLines = 2,
        )
        Button(
            onClick = onSubmit,
            enabled = enabled && draft.isNotBlank(),
            modifier = Modifier
                .fillMaxWidth()
                // Above the Material default of 48dp. Users may have limited
                // dexterity and the app is used one-handed in a home.
                .height(56.dp),
        ) {
            Text(
                text = if (enabled) "Ask" else "Working",
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

/**
 * Renders the answer, and makes the kind of answer unmistakable.
 *
 * A Redacted Answer and a Deferral are not answers. They get a different colour,
 * a different label and a different shape, so that six weeks later nobody reads
 * an old refusal as clinical advice. ADR 0004.
 */
@Composable
private fun AnswerCard(state: AskUiState) {
    if (!state.hasAnswer) return

    val (container, label) = when {
        state.answerKind == AnswerKind.DEFERRAL ->
            Color(0xFFFFF4E5) to "Not answered. Take this to your next consultation."

        state.answerKind == AnswerKind.REDACTED_ANSWER ->
            Color(0xFFFFF4E5) to "Part of this answer was held back."

        else -> Color(0xFFF1F8F2) to null
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = container),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            label?.let {
                Text(
                    text = it,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF8A5300),
                    modifier = Modifier.semantics { contentDescription = it },
                )
                Spacer(Modifier.height(8.dp))
            }
            Text(text = state.answer, style = MaterialTheme.typography.bodyLarge)
            if (state.sourceCount > 0) {
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "From ${state.sourceCount} passage(s) in the handbook",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFF5A6570),
                )
            }
            if (state.cameFromCache) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Saved answer, not a live one",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFF8A5300),
                )
            }
        }
    }
}

/**
 * Shown only for a CRITICAL emergency, which is the one severity that overrides
 * the Dose Boundary. Shown for HIGH as well would cry wolf on every bad pain
 * question.
 */
@Composable
private fun EmergencyBanner(severity: EmergencySeverity) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFDE7E7)),
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Spacer(
                Modifier
                    .size(12.dp)
                    .background(Color(0xFFC62828), RoundedCornerShape(6.dp)),
            )
            Spacer(Modifier.size(12.dp))
            Text(
                text = "This may be an emergency. Call 108 now.",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF8B1A1A),
            )
        }
    }
}

@Composable
private fun HistoryRow(question: String, answer: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = question,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(4.dp))
            Text(text = answer, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
