package commons;

import java.util.TreeMap;

/**
 * Created by tao on 8/21/17.
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;
    public TreeNode(int val){
        this.val=val;
    }
    public String toString(){
        return String.valueOf(val);
    }
}
