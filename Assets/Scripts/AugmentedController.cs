using UnityEngine;
using System.Collections;

public class AugmentedController : MonoBehaviour
{

    public Camera lookCamera;
    private CharacterController character;

    public bool rotationEnabled = false;
    public float defaultLookAngle = 0.2f;
    public float headHeight = 0.9f;
    public float forwardSpeed = 3f;
    public float minAxisTilt = 0.01f;
    public float minMovementOffset = 0.1f;
    public KeyCode rotationButtonKey = KeyCode.R;

    private bool prevVRRotateState = false;

    // Use this for initialization
    void Start()
    {
        character = GetComponent<CharacterController>();
        lookCamera.transform.localRotation = Quaternion.identity;
        UnityEngine.VR.InputTracking.Recenter();
    }

    //private bool first = true;

    public void SetWalkSpeed(float speed)
    {
        forwardSpeed = speed;
    }

    // Update is called once per frame
    void Update()
    {
        // Late start setup
        //if (first)
        //{
            //lookCamera.transform.parent.position = new Vector3(transform.position.x, transform.position.y + headHeight, transform.position.z);
            //lookCamera.transform.parent.rotation = transform.rotation;
            //lookCamera.transform.Rotate(defaultLookAngle, 0f, 0f);
            //character.Move(transform.forward * 0.0001f);
            //first = false;
        //}

        // Move forward
        float vertical = Input.GetAxis("Vertical");
        if (vertical > minAxisTilt)
        {
            Vector3 moveDirection = this.transform.forward * vertical * forwardSpeed * Time.deltaTime;
            moveDirection.y -= 9.81f * Time.deltaTime;
            character.Move(moveDirection);
        }

        // If 2D mode rotation, rotate
        if (rotationEnabled)
        {
            //float rot = UnityEngine.VR.InputTracking.GetLocalRotation(UnityEngine.VR.VRNode.CenterEye).eulerAngles.y;
            float rotBody = transform.rotation.y;
            float offset = Input.GetAxis("Horizontal");
            if (Mathf.Abs(offset) >= minMovementOffset)
            {
                float newY = offset + rotBody;
                this.transform.Rotate(0f, offset, 0f);
                //lookCamera.transform.parent.rotation = transform.rotation;
                //UnityEngine.VR.InputTracking.Recenter();
            }
        }
        else
        {
            // VR Rotation via triggers
            float trigStateRight = Input.GetAxis("RightTrigger");
            float trigStateLeft = Input.GetAxis("LeftTrigger");
            bool rotationButtonState = Input.GetKey(rotationButtonKey);
            bool vrRotateState = trigStateRight > 0 || rotationButtonState;
            bool changeOnRisingEdge = true;
            if ((changeOnRisingEdge && vrRotateState && !prevVRRotateState) || (!changeOnRisingEdge && vrRotateState))
            {
                float rotationY = UnityEngine.VR.InputTracking.GetLocalRotation(UnityEngine.VR.VRNode.CenterEye).eulerAngles.y;
                transform.rotation = Quaternion.Euler(new Vector3(0f, lookCamera.transform.rotation.eulerAngles.y, 0f));
            }
            prevVRRotateState = vrRotateState;
        }

        //lookCamera.transform.parent.position = new Vector3(transform.position.x, transform.position.y + headHeight, transform.position.z);
    }
}