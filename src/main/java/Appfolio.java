import commons.ListNode;
import commons.TreeNode;

import java.lang.reflect.Array;
import java.util.*;

public class Appfolio {


    public static void printArrays(int[]nums){
        List<Integer>duplicates = new ArrayList<>();
        List<Integer>single = new ArrayList<>();
        List<Integer>withoutDuplicate = new ArrayList<>();
        Map<Integer,Integer>map = new HashMap<>();
        for(int x:nums){
            if(!map.containsKey(x)){
                withoutDuplicate.add(x);
                map.put(x,0);
            }
            map.put(x,map.get(x)+1);
        }

        for(Map.Entry<Integer,Integer>entry:map.entrySet()){
            if(entry.getValue()>1)
                duplicates.add(entry.getKey());
            else
                single.add(entry.getKey());
        }
        System.out.println(single);
        System.out.println(duplicates);
        System.out.println(withoutDuplicate);
    }
    public void reverse(char[]ss,int begin,int end){
        while(begin<end){
            char  c = ss[begin];
            ss[begin++]=ss[end];
            ss[end--]=c;
        }
    }

    //pay attention to the trim();
    public String reverseWords(String s) {
        s = s.trim();
        char []ss = s.toCharArray();
        int n = ss.length;
        reverse(ss,0,n-1);
        StringBuilder sb = new StringBuilder();
        int i=0;
        while(i<n){
            while(i<n && ss[i]==' ')
                i++;
            int start =i;
            while(i<n && ss[i]!=' ')
                i++;
            int end =i-1;
            if(end<n){
                while(end>=start)
                    sb.append(ss[end--]);
                sb.append(' ');
            }
        }
        int nn =sb.length();
        return nn==0?"":sb.toString().substring(0,nn-1);
    }


    //without stringbuilder
    public String reverseWordsBetter(String s) {
        if (s == null) return null;

        char[] a = s.toCharArray();
        int n = a.length;

        // step 1. reverse the whole string
        reverse(a, 0, n - 1);
        // step 2. reverse each word
        reverseWords(a, n);
        // step 3. clean up spaces
        return cleanSpaces(a, n);
    }

    void reverseWords(char[] a, int n) {
        int i = 0, j = 0;

        while (i < n) {
            while (i < j || i < n && a[i] == ' ') i++; // skip spaces
            while (j < i || j < n && a[j] != ' ') j++; // skip non spaces
            reverse(a, i, j - 1);                      // reverse the word
        }
    }

    // trim leading, trailing and multiple spaces
    String cleanSpaces(char[] a, int n) {
        int i = 0, j = 0;

        while (j < n) {
            while (j < n && a[j] == ' ') j++;             // skip spaces
            while (j < n && a[j] != ' ') a[i++] = a[j++]; // keep non spaces
            while (j < n && a[j] == ' ') j++;             // skip spaces
            if (j < n) a[i++] = ' ';                      // keep only one space
        }

        return new String(a).substring(0, i);
    }



    //combination sum
    public void dfs(List<List<Integer>>res, List<Integer>path, int target, int []candidates, int ind){
        if(target==0){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=ind;i<candidates.length;++i){
            if(target>=candidates[i]){
                path.add(candidates[i]);
                dfs(res,path,target-candidates[i],candidates,i);
                path.remove(path.size()-1);
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>>res = new ArrayList<>();
        dfs(res,new ArrayList<>(),target,candidates,0);
        return res;
    }

    //给一段ruby code，oop，没用好oop，要你给他意见如何改成oop
    //LRU cache
    //implement hashmap, how to deal with collision, separate chaining, or open address
    //给你一个byte array, 一个byte 如果大于127 那么就是 double byte，否则是single byte.
    //100, 200, 10, 30 ..... 50, 100
    //倒数第二位是50，就知道这个答案是single byte， 不用从头到尾一位位看

    //merge k sorted array, 网站信息的爬虫
    //number of islands

    static class ArrayContainer {
        public List<Integer>arr;
        public int index;
        public ArrayContainer(List<Integer> arr, int index) {
            this.arr = arr;
            this.index = index;
        }
    }
    public static List<Integer>mergeKsortedArrayList(List<List<Integer>>arrays){
        PriorityQueue<ArrayContainer>pq = new PriorityQueue<>(new Comparator<ArrayContainer>() {
            @Override
            public int compare(ArrayContainer o1, ArrayContainer o2) {
                return o1.arr.get(o1.index)-o2.arr.get(o2.index);
            }
        });
        for(List<Integer>array:arrays){
            if(array!=null && !array.isEmpty())
                pq.offer(new ArrayContainer(array,0));
        }
        List<Integer>ans = new ArrayList<>();
        while(!pq.isEmpty()){
            ArrayContainer top = pq.poll();
            ans.add(top.arr.get(top.index++));
            if(top.index<top.arr.size())
                pq.offer(new ArrayContainer(top.arr,top.index));
        }
        return ans;
    }

    //reverse linkedlist
    public static ListNode reverseList(ListNode head){
        if(head==null||head.next==null)
            return head;
        ListNode newHead = null;
        ListNode p = head;
        while(p!=null){
            ListNode next = p.next;
            p.next = newHead;
            newHead = p;
            p = next;
        }
        return newHead;

    }

    public static ListNode reverseListRecursive(ListNode head){
        if(head==null||head.next==null)
            return head;
        ListNode next = reverseListRecursive(head.next);
        head.next.next = head;
        head.next = null;
        return next;
    }

    public int thirdMax(int[] nums) {
        Integer max1 = null;
        Integer max2 = null;
        Integer max3 = null;
        for (Integer n : nums) {
            if (n.equals(max1) || n.equals(max2) || n.equals(max3)) continue;
            if (max1 == null || n > max1) {
                max3 = max2;
                max2 = max1;
                max1 = n;
            } else if (max2 == null || n > max2) {
                max3 = max2;
                max2 = n;
            } else if (max3 == null || n > max3) {
                max3 = n;
            }
        }
        return max3 == null ? max1 : max3;
    }

    public boolean isSame(TreeNode root,TreeNode tree){
        if(root==null||tree==null)
            return root==tree;
        return root.val== tree.val && isSame(root.left,tree.left) && isSame(root.right,tree.right);
    }
    public boolean isSubstree(TreeNode root,TreeNode tree){
        if(isSame(root,tree))
            return true;
        return isSubstree(root.left,tree)||isSame(root.right,tree);
    }


    public static int get_restaurants(int sw_x, int sw_y, int ne_x, int ne_y){
        return 45;
    }


    //recursive way
    //不断的把长方形对半切，调用get_restaurants(sw_x, sw_y, ne_x, ne_y)，把得到的饭店放进Hashset里面，直到返回的饭店少于50为止


    public static  int get_all_restaurants(int sw_x, int sw_y, int ne_x, int ne_y){
        int val = get_restaurants(sw_x,sw_y,ne_x,ne_y);
        if(val<=50)
            return val;
        val = 0;
        int mid_x = (sw_x+ne_x)/2;
        int mid_y = (sw_y+ne_y)/2;
        return get_all_restaurants(sw_x,sw_y,mid_x,mid_y)+get_all_restaurants(sw_x,mid_y,mid_x,ne_y)+get_all_restaurants(mid_x,ne_y,ne_x,mid_y)+get_all_restaurants(mid_x,mid_y,ne_x,ne_y);
    }

    public static int get_restaurantsIterative(int sw_x,int sw_y,int ne_x,int ne_y){
        Queue<int[]>q = new LinkedList<>();
        q.offer(new int[]{sw_x,sw_y,ne_x,ne_y});
        int sum=0;
        while(!q.isEmpty()){
            int []top = q.poll();
            int val = get_restaurants(top[0],top[1],top[2],top[3]);
            if(val<=50){
                sum+=val;
            }else{
                int mid_x = (top[0]+top[2])/2;
                int mid_y = (top[1]+top[3])/2;
                q.offer(new int[]{top[0],top[1],mid_x,mid_y});
                q.offer(new int[]{top[0],mid_y,mid_x,top[3]});
                q.offer(new int[]{mid_x,top[3],top[2],mid_y});
                q.offer(new int[]{mid_x,mid_y,top[2],top[3]});
            }
        }
        return sum;
    }


    public int quickSelect(int[]nums,int begin,int end){

        int low = begin, hi=end,key =nums[low];
        while(low<hi){
            while(low<hi && nums[hi]>=key)
                hi--;
            nums[low]=nums[hi];
            while(low<hi && nums[low]<=key)
                low++;
            nums[hi]=nums[low];
        }
        nums[low]=key;
        return low;
    }
    public int findKthLargest(int[]nums,int begin,int end,int k){
        int ind = quickSelect(nums,begin,end);
        if(ind==k-1)
            return nums[k-1];
        else if(ind>k-1)
            return findKthLargest(nums,begin,ind-1,k);
        else
            return findKthLargest(nums,ind+1,end,k);
    }
    public int findKthLargest(int[] nums, int k) {
        return findKthLargest(nums,0,nums.length-1,nums.length+1-k);
    }


    public static void reverseII(char[]ss,int begin,int end){
        while(begin<end){
            char  c = ss[begin];
            ss[begin++]=ss[end];
            ss[end--]=c;
        }
    }

    public static String reverseWordsIII(String s) {
        s = s.trim();
        char []ss = s.toCharArray();
        int n = ss.length;
        reverseII(ss,0,n-1);
        StringBuilder sb = new StringBuilder();
        int i=0,start=0;;
        boolean hasWord = false;
        while(i<n||i==n){
            if(i<n && ss[i]==' '||i==n){
                if(hasWord){
                    int end =i-1;
                    if(end<n){
                        while(end>=start)
                            sb.append(ss[end--]);
                        sb.append(' ');
                    }
                    hasWord = false;
                }
                i++;
                continue;
            }
            if(!hasWord)
                start =i;
            i++;
            hasWord = true;
        }
        int nn =sb.length();
        return nn==0?"":sb.toString().substring(0,nn-1);
    }


    //merge two sorted array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int ind = m+n-1;
        m--;
        n--;
        while(m>=0 && n>=0){
            if(nums1[m]>=nums2[n]){
                nums1[ind--]=nums1[m--];
            }else
                nums1[ind--]=nums2[n--];
        }
        while(n>=0)
            nums1[ind--]=nums2[n--];
    }


    //String to Integer (atoi)
    //delete the space first
    public static int myAtoi(String str) {
        int n = str.length();
        int i=0;
        while(i<n && str.charAt(i)==' ')
            i++;
        boolean isNegative = false;
        long ans = 0;
        if(i<n && (str.charAt(i)=='+'||str.charAt(i)=='-')){
            if(str.charAt(i)=='-')
                isNegative = true;
            i++;
        }
        while(i<n && Character.isDigit(str.charAt(i))){
            ans = 10*ans+(long)(str.charAt(i++)-'0');
            if(ans>=2147483647)
                break;
        }
        ans=isNegative?-ans:ans;
        if(ans>2147483647l)
            return 2147483647;
        else if(ans<-2147483648l)
            return -2147483648;
        return (int)ans;
    }

    //Find the factorial of a number.
    public static List<Integer>factorial(int number){
        int mid = (int)Math.sqrt(number);
        List<Integer>ans = new ArrayList<>();
        for(int i=2;i<=mid;++i){
            if(number%i==0){
                if(number/i==i)
                    ans.add(i);
                else{
                    ans.add(i);
                    ans.add(number/i);
                }
            }
        }
        return ans;
    }

    public static int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int begin=0,end=0,duplicate=0,ans=0;
        int []cnt = new int[128];
        while(end<n){
            if(cnt[s.charAt(end++)]++>0)
                duplicate++;
            while(duplicate>0){
                if(cnt[s.charAt(begin++)]-- ==2)
                    duplicate--;
            }
            ans = Math.max(ans,end-begin);
        }
        return ans;
    }

    public static boolean validPassword(String password){
        boolean valid = true;
        if(password.length()<=8)
            return false;
        int n = password.length();
        char []specialCharacters = {'!','#','@','$'};
        Set<Character>specials = new HashSet<>();
        for(char c:specialCharacters)
            specials.add(c);
        boolean hasLowCase = false;
        boolean hasUpperCase = false;
        boolean hasSpecial = false;
        int cnt= 1;
        for(int i=0;i<n;++i){
            if(password.charAt(i)>='a' && password.charAt(i)<='z'){
                hasLowCase=true;
            }else if(password.charAt(i)<='Z' && password.charAt(i)>='A'){
                hasUpperCase = true;
            }else if(specials.contains(password.charAt(i))){
                hasSpecial = true;
            }
            if(i>0 && password.charAt(i)==password.charAt(i-1)){
                cnt++;
                if(cnt>=3)
                    return false;
            }else
                cnt=1;
        }
        return hasLowCase && hasUpperCase && hasSpecial;
    }

    public static void main(String []args){
//        ListNode head = new ListNode(1);
//        head.next = new ListNode(2);
//        head.next.next = new ListNode(3);
//        head.next.next.next = new ListNode(4);
//        ListNode x = reverseListRecursive(head);
//        while(x!=null){
//            System.out.println(x.val);
//            x = x.next;
//        }
//        List<Integer>array1 = new ArrayList<>(Arrays.asList(1,4,7,9,15));
//        List<Integer>array2 = new ArrayList<>(Arrays.asList(-2,5,8,12));
//        List<Integer>array3 = new ArrayList<>(Arrays.asList(-5,9,12,45));
//        List<List<Integer>> ans = new ArrayList<>();
//        ans.add(array1);
//        ans.add(array2);
//        ans.add(array3);
//        System.out.println(mergeKsortedArrayList(ans));

//        int []nums={1, 4, 2, 8, 2, 9, 1, 8, 3, 3};
//        printArrays(nums);
        //System.out.println(reverseWordsIII("the sky is blue"));
        //System.out.println(myAtoi(" "));
        //System.out.println(factorial(100));
        System.out.println(validPassword("aabAdddac@"));

    }

}
