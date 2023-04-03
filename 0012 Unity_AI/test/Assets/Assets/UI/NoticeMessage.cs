using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class NoticeMessage : MonoBehaviour
{
    [Header("SubNotice")]
    public GameObject subbox;
    public Text subintext;
    public Animator subani;

    private WaitForSeconds _UIDelay1 = new WaitForSeconds(2.0f);
    private WaitForSeconds _UIDelay2 = new WaitForSeconds(0.3f);

    // Start is called before the first frame update
    void Start()
    {
        subbox.SetActive(false);
    }

    public void Print(string message)
    {
        subintext.text = message;
        subbox.SetActive(false);
        StopAllCoroutines();
        StartCoroutine(SetDelay());
    }

    IEnumerator SetDelay()
    {
        subbox.SetActive(true);
        subani.SetBool("IsOn", true);
        yield return _UIDelay1;
        subani.SetBool("IsOn", false);
        yield return _UIDelay2;
        subbox.SetActive(false);
    }
}
