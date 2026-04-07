using UnityEngine;
using VRM;

public class ForceBlendShapeApply : MonoBehaviour
{
    public VRMBlendShapeProxy proxy;

    void LateUpdate()
    {
        if (proxy != null)
        {
            proxy.Apply(); // 🔥 FORCE update every frame
        }
    }
}