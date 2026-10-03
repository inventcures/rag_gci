package org.inventcures.pallisahayak.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One recorded interaction, as the study needs to see it.
 *
 * This is where SI-5 lands. Protocol §7.8 requires the release identifier stored
 * with every interaction, so that any answer can be traced back to the build that
 * produced it. Without that column, six months into the study nobody can answer
 * "which version gave that advice?".
 *
 * Field names match the server's vocabulary on purpose. `study_logging.py` writes
 * the same names, and keeping them aligned means an export from the phone and an
 * export from the server can be compared row by row without a translation layer.
 */
@Entity(tableName = "interactions")
data class InteractionEntity(
    @PrimaryKey
    val id: String,

    // -- who, in pseudonymised form ----------------------------------------
    // The raw participant id is never stored. The server pseudonymises with an
    // HMAC and the app does the same, so ids are linkable within a study without
    // the device holding anything that identifies a person.
    @ColumnInfo(name = "participant_id")
    val participantId: String,

    /** ASHA Worker, Family Caregiver or Patient. ADR 0006. */
    @ColumnInfo(name = "care_role")
    val careRole: String,

    @ColumnInfo(name = "site_id")
    val siteId: String,

    // -- release provenance (protocol §7.8) --------------------------------
    @ColumnInfo(name = "release_id")
    val releaseId: String,

    @ColumnInfo(name = "release_approved")
    val releaseApproved: Boolean,

    // -- what happened ------------------------------------------------------
    @ColumnInfo(name = "occurred_at")
    val occurredAt: Long,

    /** BCP-47 tag such as hi-IN. ADR 0002. */
    val language: String,

    val channel: String,

    @ColumnInfo(name = "is_offline")
    val isOffline: Boolean,

    /** True when the answer came from the cached bundle rather than the server. */
    @ColumnInfo(name = "from_cache")
    val fromCache: Boolean,

    /** Which voice path answered: live, fallback or cache. T5. */
    @ColumnInfo(name = "voice_path")
    val voicePath: String?,

    // -- the question and answer --------------------------------------------
    val query: String,

    val response: String,

    // -- safety outcome ------------------------------------------------------
    // A refusal is not an answer, and the study must never count one as guidance
    // delivered. ADR 0004.
    @ColumnInfo(name = "answer_kind")
    val answerKind: String,

    @ColumnInfo(name = "emergency_level")
    val emergencyLevel: String,

    @ColumnInfo(name = "dosage_blocked")
    val dosageBlocked: Boolean,

    @ColumnInfo(name = "source_count")
    val sourceCount: Int,

    /**
     * Protocol §4.1 counts an interaction as substantive only if it recorded a
     * real care question and returned a response or an escalation. Training
     * demonstrations, test messages and duplicate retries are excluded, so that
     * flag is stored with the row rather than reconstructed at analysis time.
     */
    @ColumnInfo(name = "interaction_class")
    val interactionClass: String,

    /** False when free text still holds a participant name. See `scrub` below. */
    @ColumnInfo(name = "text_is_scrubbed")
    val textIsScrubbed: Boolean,
)
