using UnityEngine;
using VRM;

[RequireComponent(typeof(AudioSource))]
public class YukiLipSync : MonoBehaviour
{
    public VRMBlendShapeProxy proxy;
    public float sensitivity = 100f;
    public float smoothSpeed = 10f;

    private AudioSource audioSource;
    private float currentValue = 0f;

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }

    void Update()
    {
        if (audioSource == null || proxy == null) return;

        float[] samples = new float[256];
        audioSource.GetOutputData(samples, 0);

        float volume = 0f;
        foreach (float s in samples)
            volume += Mathf.Abs(s);

        volume /= samples.Length;

        float target = volume * sensitivity;
        currentValue = Mathf.Lerp(currentValue, target, Time.deltaTime * smoothSpeed);

        currentValue = Mathf.Clamp01(currentValue);

        // Apply mouth movement
        proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.A), currentValue);
        proxy.Apply();
    }
}