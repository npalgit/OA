package design;

import commons.TreeNode;

import java.util.Deque;
import java.util.LinkedList;

/**
 * Created by tao on 9/3/17.
 */
public class StringCodec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root,sb);
        return sb.toString();
    }

    public void serialize(TreeNode root,StringBuilder sb){
        if(root==null){
            sb.append('@').append(' ');
            return;
        }
        sb.append(root.val).append(' ');
        serialize(root.left,sb);
        serialize(root.right,sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String []strs = data.split(" ");
        Deque<String> dq=new LinkedList<>();
        for(String str:strs)
            dq.add(str);
        return deserialize(dq);
    }

    public TreeNode deserialize(Deque<String>dq){
        if(dq.isEmpty())
            return null;
        String top = dq.pollFirst();
        if(top.equals("@"))
            return null;
        TreeNode node = new TreeNode(Integer.parseInt(top));
        node.left = deserialize(dq);
        node.right = deserialize(dq);
        return node;
    }
}
