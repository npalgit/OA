package commons;

/**
 * Created by tao on 10/22/17.
 */
public class MultilevelListNode {
    public int val;
    public MultilevelListNode next;
    public MultilevelListNode child;
    public MultilevelListNode(int val){
        this.val = val;
        this.next = null;
        this.child = null;
    }
}
