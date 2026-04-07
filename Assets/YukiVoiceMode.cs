using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class YukiVoiceMode : MonoBehaviour
{
    public string url = "http://127.0.0.1:8000/voice?simulated_text=hello";

    public AudioSource audioSource;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.V))
        {
            StartCoroutine(GetVoice());
        }
    }

    IEnumerator GetVoice()
    {
        UnityWebRequest request = UnityWebRequest.Get(url);
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError(request.error);
            yield break;
        }

        string json = request.downloadHandler.text;
        Debug.Log("Response: " + json);

        string audioUrl = Extract(json, "audio_url");

        yield return StartCoroutine(PlayAudio(audioUrl));
    }

    IEnumerator PlayAudio(string audioUrl)
    {
        UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(audioUrl, AudioType.WAV);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError(www.error);
            yield break;
        }

        AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
        audioSource.clip = clip;
        audioSource.Play();
    }

    string Extract(string json, string key)
    {
        string pattern = "\"" + key + "\":";
        int start = json.IndexOf(pattern);

        if (start == -1) return "";

        start += pattern.Length;

        if (json[start] == '\"')
        {
            start++;
            int end = json.IndexOf("\"", start);
            return json.Substring(start, end - start);
        }
        else
        {
            int end = json.IndexOfAny(new char[] { ',', '}' }, start);
            return json.Substring(start, end - start);
        }
    }
}