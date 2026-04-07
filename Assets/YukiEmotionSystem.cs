using UnityEngine;
using VRM;

public class YukiEmotionSystem : MonoBehaviour
{
    public VRMBlendShapeProxy proxy;

    public void SetEmotion(string emotion)
    {
        // Reset all
        proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Joy), 0);
        proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Angry), 0);
        proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Sorrow), 0);

        // Apply emotion
        if (emotion == "happy")
            proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Joy), 1f);

        else if (emotion == "angry")
            proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Angry), 1f);

        else if (emotion == "sad")
            proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Sorrow), 1f);

        // fallback for surprise
        else if (emotion == "surprised")
            proxy.SetValue(BlendShapeKey.CreateFromPreset(BlendShapePreset.Joy), 0.5f);

        proxy.Apply();
    }
}