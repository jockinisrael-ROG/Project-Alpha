using UnityEngine;

public class YukiEyeTracking : MonoBehaviour
{
    public Transform head;
    public Camera cam;

    public float lookSpeed = 5f;

    void Update()
    {
        Vector3 mousePos = Input.mousePosition;
        mousePos.z = 2f;

        Vector3 worldPos = cam.ScreenToWorldPoint(mousePos);

        // ✅ Correct base direction
        Vector3 direction = worldPos - head.position;
direction = new Vector3(-direction.x, -direction.y, direction.z);

        // ✅ ONLY flip model forward (no axis flips)
        Quaternion targetRotation = Quaternion.LookRotation(-direction);

        head.rotation = Quaternion.Slerp(
            head.rotation,
            targetRotation,
            Time.deltaTime * lookSpeed
        );
    }
}