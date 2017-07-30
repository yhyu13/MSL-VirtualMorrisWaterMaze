using UnityEngine;
using System.Collections;

public class CollisionEventTrigger : MonoBehaviour {

    public enum CollisionEventTriggerType {Start=0, Stop};
    public GameObject targetObject;
    public string tweenEventName;
    public bool triggerOnCollisionEnter = true;
    public bool triggerOnTriggerEnter = true;
    public CollisionEventTriggerType eventType = CollisionEventTriggerType.Start;
    public float collisionScore = 0;
    public float collisionReward = 0;

    void OnCollisionEnter(Collision c)
    {
        Debug.Log("OnCollisionEnter");
        if (triggerOnCollisionEnter)
        {
            float startTime = FindObjectOfType<TrialTimer>().startTime;
            float currentTime = Time.time;
            BeginNamedTweens(c.collider, targetObject, tweenEventName, eventType);
            collisionScore += 200 * Mathf.Exp((15 - (currentTime - startTime))/2); // the more time elpased, the less the score.
            collisionReward += 200 * Mathf.Exp((15 - (currentTime - startTime))/2);
        }
    }

    void OnTriggerEnter(Collider c)
    {
        Debug.Log("OnTriggerEnter");
        if (triggerOnTriggerEnter)
        {
            float startTime = FindObjectOfType<TrialTimer>().startTime;
            float currentTime = Time.time;
            BeginNamedTweens(c, targetObject, tweenEventName, eventType);
            collisionScore += 200 * (15 - (currentTime - startTime));
            collisionReward += 200 * (15 - (currentTime - startTime));
        }
    }

    static void BeginNamedTweens(Collider source, GameObject target, string name, CollisionEventTriggerType type)
    {
        Debug.Log("Source: " + source.name);
        Debug.Log("Found Matching ID");
        iTweenEditor[] targetEditors = target.GetComponents<iTweenEditor>();
        if (targetEditors == null || targetEditors.Length == 0) return;
        Debug.Log("Found Target iTweenEditor Scripts");
        foreach (iTweenEditor e in targetEditors)
        {
            if (e.name == name)
            {
                Debug.Log("Found Matching Name");
                switch (type)
                {
                    case CollisionEventTriggerType.Start:
                        Debug.Log("Starting Tween");
                        e.iTweenPlay();
                        break;
                    case CollisionEventTriggerType.Stop:
                        Debug.Log("Stopping Tween");
                        e.StopAllCoroutines();
                        break;
                }
            }
        }
    }
}
