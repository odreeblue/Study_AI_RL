using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Sockets;
using UnityEngine.SceneManagement;
//using UnityEditor.TextCore.Text;
using UnityEngine.XR;
//using System.Drawing;

//public delegate void CallbackPosAndCol(float x, float y, int isCol);// delegate 선언
public class SphereController : MonoBehaviour
{
    public float speed = 5.0f;
    public Rigidbody SphereRigidbody;
    public bool collision_flag ; // 충돌했는지 확인하는 플래
    public Vector3 move;
    public Vector3 OriginalPosition;
    public Vector3 SubjectPosition;
    public Vector3 SpherePosition;
    //public int game_done;
    public GameObject goal;
    public bool IsGoal;
    public bool IsInitial = false;
    public float bonus;
    public int max_step = 300;
    public int count_step;
    //public Vector3 LastSpherePosition;
    //public Vector3 reflect;
    void Start()
    {
        goal = GameObject.Find("Goal");
        Time.timeScale = speed;
        Debug.Log("Start CharacterMove");

        SphereRigidbody = GetComponent<Rigidbody>();
        
        collision_flag = false;
       
        SpherePosition = SphereRigidbody.transform.position;
        
        speed = 1.0f;

        move = new Vector3(0, 0, 0);
        
        Server.Instance.SetDirectionCallback(new CallbackDirection(OnDirection)); //static Server 반환됨
        IsGoal = false;

        count_step = 0;

        
    }
    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.name.Substring(0, 3) == "Out" || col.gameObject.name.Substring(0, 3) == "InS")
        {
            collision_flag = true;
            Debug.Log(col.gameObject.name);

        }
        if (col.gameObject.name.Substring(0, 2) == "Go")
        {
            //col.transform.position = new Vector3(col.transform.position.x, -2.0f,col.transform.position.z);
            IsGoal = true;

        }
       
    }
    //void OnCollisionExit(Collision col)
    //{
    //    if (col.gameObject.name.Substring(0, 3) == "Out" || col.gameObject.name.Substring(0, 3) == "InS")
    //    {
    //        collision_flag = false;
    //        Debug.Log("벽에 부딪혔다가 나옴  "+ col.gameObject.name);
    //    }

    //}

    // Update is called once per frame
    void Update()
    {
        if (IsInitial == true) // 클라이언트에서 초기값 요청하는 신호
        {
            Debug.Log("초기값 전달 ");
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.x); // position_x 전달(1)
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.z); // position_y 전달(2)
            Server.Instance.SendData.Add(99.0f); // 초기값 전달 (3)
            Server.Instance.SendData.Add(99.0f); // 초기값 전달 (4)

            // Image 데이터 전달
            String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
            byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
            Debug.Log("Image_data Size is : " + Image_data.Length); // 이미지 크기 프린트

            Server.Instance.imagedata = Image_data; // 이미지 전달(6)
            Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)
            IsInitial = false;
        }

        if (count_step < max_step)
        {
            if (move != new Vector3(0, 0, 0))// move가 입력이되었고 게임이 끝나지 않았으면
            {
                SphereRigidbody.transform.Translate(move * speed * Time.deltaTime);
            }
        }
        else
        {
            Debug.Log("max_step 초과, 다시 episode 초기화");
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.x); // position_x 전달(1)
            Server.Instance.SendData.Add(SphereRigidbody.transform.position.z); // position_y 전달(2)
            Server.Instance.SendData.Add(0.0f); // 목표 지점 도착 전에 최대 step에 도달했으니 벌점 주기(3)
            Server.Instance.SendData.Add(1.0f); // 에피소드가 끝났다고 알려주기(4)

            // Image 데이터 전달
            String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
            byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
            Debug.Log("Image_data Size is : " + Image_data.Length); // 이미지 크기 프린트
            
            Server.Instance.imagedata = Image_data; // 이미지 전달(6)
            Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)


            move = new Vector3(0, 0, 0);
            count_step = 0;
            SphereRigidbody.transform.position = new Vector3(4.0f, 3.0f, -4.0f); // 공 위치 초기화
            collision_flag = false;
            SpherePosition = new Vector3(4.0f, 0.0f, -4.0f);
        }

    }
    void LateUpdate()
    {
        if(move != new Vector3(0, 0, 0))
        {
            float epsilon = 0.1f;
            float loss_x = SphereRigidbody.transform.position.x - SubjectPosition.x;
            //float loss_y = SphereRigidbody.transform.position.y - SubjectPosition.y;
            float loss_z = SphereRigidbody.transform.position.z - SubjectPosition.z;

            float pos_y = SphereRigidbody.transform.position.y;

            float abs_xz = Math.Abs(loss_x) + Math.Abs(loss_z);
            Debug.Log("SphereRigidbody.x: " + SphereRigidbody.transform.position.x + "-SubjectPosition.x :" + SubjectPosition.x + " ,loss x : " + loss_x);
            Debug.Log("SphereRigidbody.z: " + SphereRigidbody.transform.position.z + "-SubjectPosition.z :" + SubjectPosition.z + " ,loss z : " + loss_z);
            //Debug.Log("loss z : " + loss_z);
            Debug.Log("total abs loss : " + abs_xz + "  Collision_flag " + collision_flag + "  move : " + move[0] + "," + move[1] + "," + move[2]);
            // 명령된 위치에 거의(0.1 이하 수준) 도달 & 충돌 없음
            if ((abs_xz <= epsilon) && collision_flag == false && move != new Vector3(0, 0, 0))// && pos_y > -0.1)
            {
                Server.Instance.SendData.Add(OriginalPosition.x); // position_x 전달(1)
                Server.Instance.SendData.Add(OriginalPosition.z); // position_y 전달(2)


                if (IsGoal == true) // 골 지점에 도달했다면
                {
                    
                    Server.Instance.SendData.Add(1.0f); // 목표지점에 도착했으니 보상 주기(3)
                    Server.Instance.SendData.Add(1.0f); // 에피소드가 끝났다고 전달하기(4)
                    // Image 데이터 전달
                    String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
                    byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
                    Debug.Log("Image_data Size is : "+ Image_data.Length); // 이미지 크기 프린트
                    
                    Server.Instance.imagedata = Image_data; // 이미지 전달(6)
                    Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)

                    SphereRigidbody.transform.position = new Vector3(4.0f, 3.0f, -4.0f); // 공 위치 초기화
                    SpherePosition = new Vector3(4.0f, 0.0f, -4.0f); // 
                    IsGoal = false;
                    count_step = 0;
                    
                }
                else // 골 지점에 도달하지 않았다
                {
                    Server.Instance.SendData.Add(0.01f); // 목표지점에 도착하지 않았고, 벽에 부딪히지 않았으니 0.01점 보상(3)
                    Server.Instance.SendData.Add(0.0f); // 에피소드가 끝나지 않았다고 전달하기(4)
                    // Image 데이터 전달
                    String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
                    byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
                    Debug.Log("Image_data Size is : " + Image_data.Length); // 이미지 크기 프린트
                    
                    Server.Instance.imagedata = Image_data; // 이미지 전달(6)
                    Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)

                    SpherePosition = SphereRigidbody.transform.position;
                    
                }
                //bonus = 0.0f;
                move = new Vector3(0, 0, 0);

            }
            // 충돌 있음
            else if (collision_flag == true && move != new Vector3(0, 0, 0))// && pos_y > -0.1)
            {
                Server.Instance.SendData.Add(OriginalPosition.x); // position_x 전달(1)
                Server.Instance.SendData.Add(OriginalPosition.z); // position_y 전달(2)
                Server.Instance.SendData.Add(0.0f); // 목표 지점 도착 전에 벽에 충돌했으니 벌점 주기(3)
                Server.Instance.SendData.Add(0.0f); // 에피소드가 끝나지 않았다고 전달하기(4)
                move = new Vector3(0, 0, 0);
                SphereRigidbody.transform.position = OriginalPosition; // 공 위치 초기화

                // Image 데이터 전달
                String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
                byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
                Debug.Log("Image_data Size is : " + Image_data.Length); // 이미지 크기 프린트
                
                Server.Instance.imagedata = Image_data; // 이미지 전달(6)
                Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)


                collision_flag = false;
                SpherePosition = OriginalPosition;
                //SpherePosition = new Vector3(4.0f, 0.0f, -4.0f);


            }
           
        }
        
    }

    void OnDirection(int direction)
    {
        if(direction == 99) // 클라이언트에서 초기값 요청
        {
            IsInitial = true;
            //Debug.Log("초기값 전달 ");
            //Server.Instance.SendData.Add(SphereRigidbody.transform.position.x); // position_x 전달(1)
            //Server.Instance.SendData.Add(SphereRigidbody.transform.position.z); // position_y 전달(2)
            //Server.Instance.SendData.Add(99.0f); // 초기값 전달 (3)
            //Server.Instance.SendData.Add(99.0f); // 초기값 전달 (4)

            //// Image 데이터 전달
            //String Image_data = Capture.Instance.ScreenShot(); //스크린샷 찍고
            //byte[] Length = Convert.FromBase64String(Image_data); // 이미지 크기 확인
            //Debug.Log("Image_data Size is : " + Image_data.Length); // 이미지 크기 프린트

            //Server.Instance.imagedata = Image_data; // 이미지 전달(6)
            //Server.Instance.SendData.Add(Image_data.Length); // 이미지 크기 전달(5)
        }
        else
        {
            count_step += 1;

            Debug.Log("Step : " + count_step + ", CharacterMove : " + direction);

            switch (direction)
            {
                case 0: //위쪽 방향 -z방향
                    move = new Vector3(0, 0, -1f) * 0.5f;
                    break;
                case 1: //아래쪽 방향 z방향
                    move = new Vector3(0, 0, 1f) * 0.5f;
                    break;
                case 2: //오른쪽 방향 -x 방향
                    move = new Vector3(-1f, 0, 0) * 0.5f;
                    break;
                case 3: // 왼쪽 방향 x 방향
                    move = new Vector3(1f, 0, 0) * 0.5f;
                    break;
            }
            //this.game_done = done;
            OriginalPosition = SpherePosition;
            SubjectPosition = SpherePosition + move;
        }
        

    }
}
