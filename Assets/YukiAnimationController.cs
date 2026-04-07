using UnityEngine;
using System.Collections;
using System.Diagnostics;
using System.IO;

public class YukiController : MonoBehaviour
{
    public Animator animator;
    private Camera cam;
    private AudioSource audioSource;

    public SkinnedMeshRenderer faceMesh;

    int indexA;
    float[] samples = new float[256];

    private float mouseDownTime;
    private bool isDragging = false;
    private float clickThreshold = 0.15f;

    private bool isBusy = false;

    private float idleTimer = 0f;
    private float idleInterval = 20f;

    public float followSpeed = 20f;
    public float swayAmount = 15f;
    public float swaySmooth = 10f;

    private Vector3 lastPos;
    private Vector3 dragOffset;

    // 🔥 VOICE LINES
    string[] voiceLines =
    {
        "Hey!! You wanna fight?!",
        "Come on, don't hold back!",
        "Hah! Is that all you got?",
        "You're pretty good... for a human!",
        "Hey! You think you can beat me? Try it!!!"
    };

    void Start()
    {
        animator = GetComponent<Animator>();
        cam = Camera.main;

        audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.spatialBlend = 0f;
        audioSource.volume = 1f;

        if (faceMesh != null)
        {
            indexA = faceMesh.sharedMesh.GetBlendShapeIndex("Fcl_MTH_A");
        }

        CrossFade("Idle_Drunk", 0.2f);
        lastPos = transform.position;
    }

    void Update()
    {
        if (!isBusy && !isDragging)
        {
            HandleIdle();
        }

        HandleLipSync();
    }

    void OnMouseDown()
    {
        mouseDownTime = Time.time;
        isDragging = false;
        dragOffset = transform.position - GetMouseWorldPos();
    }

    void OnMouseDrag()
    {
        if (Time.time - mouseDownTime > clickThreshold)
        {
            if (!isDragging)
            {
                isDragging = true;
                isBusy = true;

                animator.speed = 2f;
                CrossFade("Hang", 0.1f);
            }

            Vector3 target = GetMouseWorldPos() + dragOffset;

            transform.position = Vector3.Lerp(
                transform.position,
                target,
                Time.deltaTime * followSpeed
            );

            Vector3 delta = transform.position - lastPos;

            float tilt = Mathf.Clamp(delta.x * swayAmount, -20f, 20f);
            Quaternion targetRot = Quaternion.Euler(0, 0, -tilt);

            transform.rotation = Quaternion.Lerp(
                transform.rotation,
                targetRot,
                Time.deltaTime * swaySmooth
            );

            lastPos = transform.position;
        }
    }

    void OnMouseUp()
    {
        float pressTime = Time.time - mouseDownTime;

        animator.speed = 1f;

        if (isDragging)
        {
            isDragging = false;

            StopAllCoroutines();
            isBusy = false;

            transform.rotation = Quaternion.identity;

            animator.Rebind();
            animator.Update(0f);

            animator.Play("Idle_Drunk", 0, 1f);

            idleTimer = 0f;
            return;
        }

        if (pressTime < clickThreshold && !isBusy)
        {
            StartCoroutine(PlayRandomVoice());
            StartCoroutine(PlayOnce("Boxing_Idle", 2.5f));
        }
    }

    void HandleIdle()
    {
        idleTimer += Time.deltaTime;

        if (idleTimer >= idleInterval)
        {
            idleTimer = 0f;

            int rand = Random.Range(0, 2);

            if (rand == 0)
                StartCoroutine(PlayOnce("Idle_Float", 2.5f));
            else
                StartCoroutine(PlayOnce("Idle_Salsa", 3f));
        }
    }

    void CrossFade(string anim, float time)
    {
        animator.CrossFade(anim, time, 0);
    }

    IEnumerator PlayOnce(string anim, float duration)
    {
        if (isBusy) yield break;

        isBusy = true;

        CrossFade(anim, 0.15f);

        yield return new WaitForSeconds(duration);

        CrossFade("Idle_Drunk", 0.25f);

        yield return new WaitForSeconds(0.2f);

        isBusy = false;
        idleTimer = 0f;
    }

    // 🔥 VOICE SYSTEM (TTS + FIXED AUDIO)
IEnumerator PlayRandomVoice()
{
    int index = Random.Range(0, voiceLines.Length);
    string text = voiceLines[index];

    string folder = Application.dataPath;

    // 🔥 CLEAN OLD FILES (IMPORTANT)
    string[] oldFiles = Directory.GetFiles(folder, "voice_*.mp3");
    foreach (string file in oldFiles)
    {
        try { File.Delete(file); } catch {}
    }

    string id = System.DateTime.Now.Ticks.ToString();
    string mp3Path = folder + "/voice_" + id + ".mp3";

    int attempts = 0;
    bool success = false;

    while (attempts < 3 && !success)
    {
        attempts++;

        string command = $"python -m edge_tts --voice en-US-AnaNeural --text \"{text}\" --write-media \"{mp3Path}\"";

        var process = new System.Diagnostics.Process();
        process.StartInfo.FileName = "cmd.exe";
        process.StartInfo.Arguments = "/C " + command;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();

        while (!process.HasExited)
            yield return null;

        // 🔥 CHECK FILE VALIDITY
        if (File.Exists(mp3Path))
        {
            FileInfo fi = new FileInfo(mp3Path);

            if (fi.Length > 5000) // must not be empty
            {
                success = true;
                break;
            }
        }

        UnityEngine.Debug.LogWarning("TTS retry...");
        yield return new WaitForSeconds(0.5f);
    }

    if (!success)
    {
        UnityEngine.Debug.LogError("TTS completely failed");
        yield break;
    }

    // 🔥 LOAD AUDIO
    AudioClip clip = null;

    using (WWW www = new WWW("file://" + mp3Path))
    {
        yield return www;

        if (!string.IsNullOrEmpty(www.error))
        {
            UnityEngine.Debug.LogError(www.error);
            yield break;
        }

        clip = www.GetAudioClip(false, false, AudioType.MPEG);
    }

    if (clip == null)
    {
        UnityEngine.Debug.LogError("Clip invalid");
        yield break;
    }

    audioSource.clip = clip;
    audioSource.Play();

    // 🔥 AUTO DELETE AFTER PLAY
    StartCoroutine(DeleteAfterPlay(mp3Path, clip.length));
}
    // 🔥 FINAL LIP SYNC (REALISTIC)
    void HandleLipSync()
    {
        if (faceMesh == null || audioSource == null) return;

        float target = 0f;

        if (audioSource.isPlaying && audioSource.clip != null)
        {
            float[] data = new float[256];
            audioSource.GetOutputData(data, 0);

            float sum = 0f;

            for (int i = 0; i < data.Length; i++)
            {
                sum += data[i] * data[i];
            }

            float rms = Mathf.Sqrt(sum / data.Length);

            target = Mathf.Clamp(rms * 12000f, 0f, 100f);
        }

        float current = faceMesh.GetBlendShapeWeight(indexA);
        float speed = audioSource.isPlaying ? 20f : 40f;

        float smooth = Mathf.Lerp(current, target, Time.deltaTime * speed);

        faceMesh.SetBlendShapeWeight(indexA, smooth);
    }
    IEnumerator WaitForFileReady(string path)
{
    float timer = 0f;

    while (true)
    {
        if (File.Exists(path))
        {
            try
            {
                using (FileStream stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.None))
                {
                    if (stream.Length > 0)
                        break;
                }
            }
            catch {}
        }

        timer += Time.deltaTime;
        if (timer > 5f)
        {
            UnityEngine.Debug.LogError("File never became ready");
            break;
        }

        yield return null;
    }
}
IEnumerator DeleteAfterPlay(string path, float delay)
{
    yield return new WaitForSeconds(delay + 1f);

    if (File.Exists(path))
    {
        try { File.Delete(path); } catch {}
    }
}

    Vector3 GetMouseWorldPos()
    {
        Vector3 mouse = Input.mousePosition;

        float z = cam.WorldToScreenPoint(transform.position).z;

        Vector3 world = cam.ScreenToWorldPoint(new Vector3(mouse.x, mouse.y, z));

        world.z = transform.position.z;

        return world;
    }
}