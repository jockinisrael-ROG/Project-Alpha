using UnityEngine;

public class IdleController : MonoBehaviour
{
    float timer;

    void Update()
    {
        timer += Time.deltaTime;

        if (timer > Random.Range(2f, 5f))
        {
            Debug.Log("Blink");
            timer = 0f;
        }
    }
}