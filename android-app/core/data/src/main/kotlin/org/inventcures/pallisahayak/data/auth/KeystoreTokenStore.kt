package org.inventcures.pallisahayak.data.auth

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import org.json.JSONException
import org.json.JSONObject
import java.security.GeneralSecurityException
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Keeps the session encrypted with a key that never leaves the Android Keystore.
 *
 * The tokens are what let a phone ask questions as a participant, so they are not
 * written in the clear. The session is serialised to JSON, encrypted with
 * AES-GCM under a Keystore key, and stored in private preferences. The app's
 * manifest already turns backup off, so the ciphertext is not copied off the
 * phone either, and the key could not decrypt it anywhere else.
 *
 * Written against the Keystore directly rather than androidx.security, which is
 * deprecated and pulls in Tink, a large dependency for one encrypted record on
 * a size-budgeted build.
 */
class KeystoreTokenStore(context: Context) : TokenStore {

    private val prefs = context.applicationContext
        .getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    override fun load(): StoredSession? {
        val sealed = prefs.getString(KEY_SESSION, null) ?: return null
        return try {
            decode(decrypt(sealed))
        } catch (error: GeneralSecurityException) {
            // The key is gone (a restore, or the Keystore was reset). The record
            // cannot be read, so the phone starts again at registration.
            Log.w(TAG, "stored session could not be decrypted; clearing it", error)
            clear()
            null
        } catch (error: JSONException) {
            Log.w(TAG, "stored session was malformed; clearing it", error)
            clear()
            null
        } catch (error: IllegalArgumentException) {
            Log.w(TAG, "stored session was not valid base64; clearing it", error)
            clear()
            null
        }
    }

    override fun save(session: StoredSession) {
        prefs.edit().putString(KEY_SESSION, encrypt(encode(session))).apply()
    }

    override fun clear() {
        prefs.edit().remove(KEY_SESSION).apply()
    }

    private fun encrypt(plain: String): String {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val sealed = cipher.iv + cipher.doFinal(plain.toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(sealed, Base64.NO_WRAP)
    }

    private fun decrypt(encoded: String): String {
        val sealed = Base64.decode(encoded, Base64.NO_WRAP)
        if (sealed.size <= IV_BYTES) throw GeneralSecurityException("stored session is truncated")
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(
            Cipher.DECRYPT_MODE,
            key(),
            GCMParameterSpec(TAG_BITS, sealed.copyOfRange(0, IV_BYTES)),
        )
        return String(cipher.doFinal(sealed.copyOfRange(IV_BYTES, sealed.size)), Charsets.UTF_8)
    }

    private fun key(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (keyStore.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(KEY_BITS)
                .build(),
        )
        return generator.generateKey()
    }

    private fun encode(session: StoredSession): String = JSONObject()
        .put("user_id", session.userId)
        .put("care_role", session.careRole)
        .put("site_id", session.siteId)
        .put("language", session.language)
        .putOpt("access_token", session.accessToken)
        .putOpt("refresh_token", session.refreshToken)
        .putOpt("access_expires_at", session.accessExpiresAt)
        .toString()

    private fun decode(json: String): StoredSession {
        val record = JSONObject(json)
        return StoredSession(
            userId = record.getString("user_id"),
            careRole = record.getString("care_role"),
            siteId = record.getString("site_id"),
            language = record.getString("language"),
            accessToken = record.optStringOrNull("access_token"),
            refreshToken = record.optStringOrNull("refresh_token"),
            accessExpiresAt = if (record.has("access_expires_at")) record.getDouble("access_expires_at") else null,
        )
    }

    private fun JSONObject.optStringOrNull(name: String): String? =
        if (has(name) && !isNull(name)) getString(name) else null

    private companion object {
        const val TAG = "TokenStore"
        const val PREFS = "palli_session"
        const val KEY_SESSION = "session"
        const val KEYSTORE = "AndroidKeyStore"
        const val ALIAS = "palli_session_key"
        const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val IV_BYTES = 12
        const val TAG_BITS = 128
        const val KEY_BITS = 256
    }
}
