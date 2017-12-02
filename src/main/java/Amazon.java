import commons.Point;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by tao on 10/3/17.
 */


public class Amazon {


    //overlap
    boolean doOverlap(Point l1, Point r1, Point l2, Point r2)
    {
        // If one rectangle is on left side of other
        if (l1.x >= r2.x || l2.x >= r1.x)
            return false;

        // If one rectangle is above other
        if (l1.y <= r2.y || l2.y <= r1.y)
            return false;

        return true;
    }

    public Point[] Solution(Point[] array, Point origin, int k) {
        Point[] rvalue = new Point[k];
        int index = 0;
        PriorityQueue<Point> pq = new PriorityQueue<Point>(k, new Comparator<Point>() {
            @Override
            public int compare(Point a, Point b) {
                return (int) (getDistance(a, origin) - getDistance(b, origin));
            }
        });

        for (int i = 0; i < array.length; i++) {
            pq.offer(array[i]);
            if (pq.size() > k)
                pq.poll();
        }
        while (!pq.isEmpty())
            rvalue[index++] = pq.poll();
        return rvalue;
    }
    private double getDistance(Point a, Point b) {
        return Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    public List<Integer> GetSum(List<Integer> A, int k) {
        ArrayList<Integer> result  = new ArrayList<>();
        if (A == null || A.size() == 0 || k <= 0) return result;
        int n= A.size(),sum=0;
        for (int i = 0; i <n; i++){
            sum+=A.get(i);
            if(i>=k-1){
                result.add(sum);
                sum-=A.get(i-k+1);
            }
        }
        return result;
    }

    public int[] getSum(int[] array, int k) {
        if (array == null || array.length < k || k <= 0)    return null;
        int[] rvalue = new int[array.length - k + 1];
        for (int i = 0; i < k; i++)
            rvalue[0] += array[i];
        for (int i = 1; i < rvalue.length; i++) {
            rvalue[i] = rvalue[i-1] - array[i-1] + array[i+k-1];
        }
        return rvalue;
    }

    public String longestPalindrome(String s) {
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
                if(dp[i][j] && j-i+1>len){
                    len = j-i+1;
                    start = i;
                }
            }
        }
        return s.substring(start,start+len);
    }

    public int extendMiddle(char[]ss,int mid){
        int left = mid-1;
        int right=mid+1;
        int end = ss.length;
        while(left>=0 && right<end){
            if(ss[left]!=ss[right])
                break;
            left--;
            right++;
        }
        return right-left-1;
    }
    public int extend(char[]ss,int middle){
        int left=middle,right=middle+1;
        int n = ss.length;
        while(left>=0 && right<n){
            if(ss[left]!=ss[right])
                break;
            left--;
            right++;
        }
        return right-left-1;

    }
    public String longestPalindromeExtend(String s) {
        //extend
        char []ss =s.toCharArray();
        int n = s.length(),start=0,len=0;
        for(int i=0;i<n;++i){
            int length = extendMiddle(ss,i);
            if(length>len){
                len=length;
                start=i-length/2;
            }
            length=extend(ss,i);
            if(length>len){
                len=length;
                start=i-length/2+1;
            }
        }
        return s.substring(start,start+len);
    }

    public String longestPalindromeM(String s) {
        String T = preProcess(s);
        int n = T.length();
        int[] p = new int[n];
        int center = 0, right = 0;
        for (int i = 1; i < n - 1; i++) {
            int j = 2 * center - i;  //j and i are symmetric around center
            p[i] = (right > i) ? Math.min(right - i, p[j]) : 0;

            // Expand palindrome centered at i
            while (T.charAt(i + 1 + p[i]) == T.charAt(i - 1 - p[i]))
                p[i]++;

            // If palindrome centered at i expand past right,
            // then adjust center based on expand palindrome
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
        }

        //  Find the longest palindrome
        int maxLength = 0, centerIndex = 0;
        for (int i = 1; i < n - 1; i++) {
            if (p[i] > maxLength) {
                maxLength = p[i];
                centerIndex = i;
            }
        }

        centerIndex = (centerIndex - 1 - maxLength) / 2;
        return s.substring(centerIndex, centerIndex + maxLength);
    }

    // preProcess the original string s.
    // For example, s = "abcdefg", then the rvalue = "^#a#b#c#d#e#f#g#$"
    private String preProcess(String s) {
        if (s == null || s.length() == 0)  return "^$";
        StringBuilder rvalue = new StringBuilder("^");
        for (int i = 0; i < s.length(); i++)
            rvalue.append("#").append(s.substring(i, i+1));
        rvalue.append("#$");
        return rvalue.toString();
    }

}
