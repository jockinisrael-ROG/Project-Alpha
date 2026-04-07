using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

[System.Serializable]
public class BackendResponse
{
    public string text;
    public string emotion;
    public string audio_url;
}

public class YukiBackendConnector : MonoBehaviour
{
    public string backendURL = "http://127.0.0.1:8000/voice";

    public AudioSource audioSource;
    public YukiEmotionSystem emotionSystem;

    void Update()
    {
        // 🔥 PRESS V TO TALK
        if (Input.GetKeyDown(KeyCode.V))
        {
            SendVoice();
        }
    }

    public void SendVoice()
    {
        Debug.Log("📡 Sending request...");
        StartCoroutine(SendRequest());
    }

    IEnumerator SendRequest()
    {
        // 🔥 IMPORTANT JSON
        string json = "{\"text\": \"\"}";

        UnityWebRequest req = new UnityWebRequest(backendURL, "POST");

        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();

        req.SetRequestHeader("Content-Type", "application/json");
        req.method = UnityWebRequest.kHttpVerbPOST;

        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("❌ Backend Error: " + req.error);
            Debug.LogError("❌ Response: " + req.downloadHandler.text);
            yield break;
        }

        Debug.Log("✅ Response: " + req.downloadHandler.text);

        BackendResponse res = JsonUtility.FromJson<BackendResponse>(req.downloadHandler.text);

        if (res == null)
        {
            Debug.LogError("❌ JSON parse failed!");
            yield break;
        }

        // 🎭 Emotion
        if (emotionSystem != null)
            emotionSystem.SetEmotion(res.emotion);

        // 🔊 Audio
        if (!string.IsNullOrEmpty(res.audio_url))
            StartCoroutine(PlayAudio(res.audio_url));
    }

    IEnumerator PlayAudio(string url)
    {
        Debug.Log("🔊 Loading audio...");

        UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV);

        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("❌ Audio Error: " + www.error);
            yield break;
        }

        AudioClip clip = DownloadHandlerAudioClip.GetContent(www);

        if (clip == null)
        {
            Debug.LogError("❌ Audio clip NULL!");
            yield break;
        }

        audioSource.clip = clip;
        audioSource.Play();

        Debug.Log("▶️ Audio playing");
    }
}