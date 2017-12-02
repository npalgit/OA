package medium;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

/**
 * Created by tao on 9/14/17.
 */
public class PureStorage {

    //7

    public int longestPalindrome(String s) {
        //o(n^2) dp
        int m = s.length();
        boolean [][]dp=new boolean[m][m];
        for(int i=0;i<m;++i){
            dp[i][i]=true;
            for(int j=0;j<i;++j){
                if(s.charAt(i)==s.charAt(j) && (i-1<j+1||dp[j+1][i-1])){
                    dp[j][i]=true;
                }
            }
        }
        int len=0,start=0,cnt=0;
        for(int i=0;i<m;++i){
            for(int j=i;j<m;++j){
                if(dp[i][j]){
                    cnt++;
                }
            }
        }
        return cnt;
    }


    //8 acquire lock and unlock
    public int detect(String[]inputs){
        Set<Integer>set = new HashSet<>();
        Stack<Integer> stk = new Stack<>();
        int size = inputs.length;
        for(int i=0;i<size;++i){
            String [] ss = inputs[i].split(" ");
            String status = ss[0];
            int id = Integer.parseInt(ss[1]);
            if(status.equals("ACQUIRE")){
                if(set.contains(id))
                    return i+1;
                else{
                    stk.push(id);
                    set.add(id);
                }
            }else{
                if(stk.isEmpty()||stk.pop()!=id){
                    return i+1;
                }else
                    set.remove(id);
            }
        }
        return stk.isEmpty()?0:1+size;
    }


    public int sorted_search(int[]elements,int target){
        if(elements==null||elements.length<=0)
            return -1;
        int left =0, right = elements.length-1;
        while(left<right){
            int mid = (left+right+1)/2;
            if(elements[mid]>target)
                right=mid-1;
            else
                left=mid+1;
        }
        return elements[right]==target?right:-1;
    }
}
