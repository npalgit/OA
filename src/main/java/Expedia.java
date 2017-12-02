import commons.ListNode;
import commons.TreeNode;

import java.util.*;
import java.math.*;

/**
 * Created by tao on 9/29/17.
 */
public class Expedia {

    public int[]sortByFrequence(int []nums){
        int n = nums.length;
        List<Integer>[]freq =  new ArrayList[n+1];
        Map<Integer,Integer> map =new HashMap<>();
        for(int x:nums)
            map.put(x,map.getOrDefault(x,0)+1);
        for(Map.Entry<Integer,Integer>entry:map.entrySet()){
            if(freq[entry.getValue()]==null)
                freq[entry.getValue()]=new ArrayList<>();
            freq[entry.getValue()].add(entry.getKey());
        }
        for(int i=1;i<=n;++i)
            if(freq[i]!=null && freq[i].size()>1)
                Collections.sort(freq[i]);
        int []res = new int[n];
        int ind =0 ;
        for(int i=1;i<=n;++i){
            if(freq[i]!=null)
                for(int x:freq[i]){
                    for(int j=0;j<i;++j)
                        res[ind++]=x;
                }
        }
        return res;
    }

    //find strings
    public List<String>getString(String[]args1,String[]args2){
        //set
        Set<String>sets = new HashSet<>();
        for(String str:args2)
            sets.add(str);
        int m = args1.length;
        List<String>res = new ArrayList<>();
        for(int i=0;i<m;++i){
            if(!sets.contains(args1[i]))
                res.add(args1[i]);
        }
        return res;
    }

    //find the 4th bit
    public int getBits(int x){
        return x<8?0:(x>>3)&0x1;
    }

    public int subCount(int[]nums,int k){
        int n = nums.length;
        int []cnt = new int[k];
        int cntNum = 0, sum =0;
        for(int i=0;i<n;++i){
            sum +=nums[i];
            int rem = sum%k;
            if(rem==0){
                cntNum+=++cnt[0];
            }else{
                cntNum+=cnt[rem];
                cnt[rem]++;
            }
        }
        return cntNum;
    }


    public int inTree(TreeNode root,int val){
        if(root==null)
            return 0;
        if(root.val==val)
            return 1;
        if(root.val>val)
            return inTree(root.left,val);
        else
            return inTree(root.right,val);
    }

    public int getRoman(String roman){
        char []ss = roman.toCharArray();
        int n =ss.length;
        Map<Character,Integer>map=new HashMap<>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);
        int sum = map.get(ss[n-1]);
        int ind =n-2;
        while(ind>=0){
            if(map.get(ss[ind+1])<map.get(ss[ind]))
                sum+=map.get(ss[ind]);
            else
                sum-=map.get(ss[ind]);
            ind--;
        }
        return sum;
    }
    public String[]sortName(String[]args){
        Arrays.sort(args, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String []args1 = o1.split(" ");
                String []args2 = o2.split(" ");
                if(!args1[0].equals(args2[0]))
                    return args1[0].compareTo(args2[0]);
                else{
                    return getRoman(args2[1])-getRoman(args1[1]);
                }
            }
        });
        return args;
    }

    public int adjustNumber(int[]nums){
        int minVal = Integer.MAX_VALUE,n=nums.length;
        int sum = 0;
        for(int x:nums){
            sum+=x;
            if(minVal>x)
                minVal=x;
        }
        return sum-minVal*n;

    }

    public ListNode removeOdd(ListNode head){
        if(head==null||head.next==null)
            return null;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p = dummy;
        while(p!=null && p.next!=null){
            p.next=p.next.next;
            p=p.next;
        }
        if(p!=null)
            p.next=null;
        return dummy.next;
    }

    //481 magic string
    //judge point in a triangle
    //cross product
    public int getArea(int x1,int y1,int x2,int y2,int x3,int y3){
        //(x1-x2,y1-y2)x(x1-x3,y1-y3);
        return Math.abs((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3));
    }
    public boolean isInside(int x1,int y1,int x2,int y2,int x3,int y3,int x4,int y4){
        int area = getArea(x1,y1,x2,y2,x3,y3);
        int area1 = getArea(x1,y1,x2,y2,x4,y4);
        int area2 = getArea(x1,y1,x4,y4,x3,y3);
        int area3 = getArea(x4,y4,x2,y2,x3,y3);
        return area==(area1+area3+area2);
    }

    //string compression
    //output size should be smaller than input's size
    public String compression(String str){
        char []ss = str.toCharArray();
        int n = ss.length;
        int start=0,ind=1;
        StringBuilder sb = new StringBuilder();
        while(ind<=n){
            if(ind==n||ss[ind]!=ss[ind-1]){
                int len = ind-start;
                start=ind;
                if(len>1){
                    sb.append(ss[ind-1]).append(len);
                }else
                    sb.append(ss[ind-1]);
            }
            ind++;
        }
        return sb.toString();
    }

    //第一个题parse，很简单。第二个题counting group，给一个只有0和1的matrix
    public int countGroup(int[][]matrix){
        if(matrix.length==0||matrix[0].length==0)
            return 0;
        int m = matrix.length,n = matrix[0].length;
        boolean [][]vis = new boolean[m][n];
        int []dx={1,-1,0,0};
        int []dy ={0,0,1,-1};
        Map<Integer,Integer>map = new HashMap<>();
        Queue<int[]>q=new LinkedList<>();
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(matrix[i][j]==1 && !vis[i][j]){
                    q.offer(new int[]{i,j});
                    vis[i][j]=true;
                    int size=0;
                    while(!q.isEmpty()){
                        int []top = q.poll();
                        size++;
                        int x = top[0];
                        int y =top[1];
                        for(int k=0;k<4;++k){
                            int nx = x+dx[k];
                            int ny = y+dy[k];
                            if(nx<0||nx>=m||ny<0||ny>=n||vis[nx][ny]||matrix[nx][ny]!=1)
                                continue;
                            vis[nx][ny]=true;
                            q.offer(new int[]{nx,ny});
                        }
                    }
                    map.put(size,map.getOrDefault(size,0)+1);
                }

            }
        }
        return 1;
    }





    //计算对应的size的group的数目，比如问元素数目为5的group有多少个，用bfs解决。
    //可以写一写

    //lc 47 permutationII


    //one dimension Dungeon Game
    public int calculateMinimumHP(int[]dungeon) {
        if(dungeon.length==0)
            return 1;
        int n = dungeon.length;
        int []dp=new int[n];
        dp[n-1]=Math.max(1,1-dungeon[n-1]);
        for(int i=n-2;i>=0;--i){
            dp[i]=Math.max(1,dp[i+1]-dungeon[i]);
        }
        return dp[0];
    }


    //598. Range Addition II
    public int maxCount(int m, int n, int[][] ops) {
        for(int []op:ops){
            m=Math.min(m,op[0]);
            n=Math.min(n,op[1]);
        }
        return m*n;
    }
    //balanced sales;
    public int balancedSales(int[]sales){
        int sum = 0;
        for(int sale:sales)
            sum+=sale;
        int n = sales.length,cur=0;
        for(int i=0;i<n;++i){
            if(cur==sum-cur-sales[i])
                return i;
            cur+=sales[i];
        }
        return 0;
    }

    //separating students
    //和sort color问题很像
    public int minimum(int[]nums,boolean zeroFirst){
        int begin =0, end = nums.length-1,cur=0;
        while(begin<end){
            if(zeroFirst){
                while(begin<end && nums[begin]==0)
                    begin++;
                while(begin<end && nums[end]==1)
                    end--;
            }else{
                while(begin<end && nums[begin]==1)
                    begin++;
                while(begin<end && nums[end]==0)
                    end--;
            }
            if(begin<end){
                int tmp = nums[begin];
                nums[begin++]=nums[end];
                nums[end--]=tmp;
                cur++;
            }

        }
        return cur;
    }
    public int separatingStudent(int[]nums){
        int begin =0,end=nums.length-1;
        int []copy = nums.clone();
        return Math.min(minimum(nums,true),minimum(copy,false));
    }

    public List<String>missingString(String s, String t){
        List<String>res = new ArrayList<>();
        String []tt = t.split(" ");
        String []ss = s.split(" ");
        Map<String,Integer>map = new HashMap<>();
        for(String str:tt){
            map.put(str,map.getOrDefault(str,0)+1);
        }
        for(String str:ss){
            if(!map.containsKey(str)||map.get(str)<1)
                res.add(str);
            else
                map.put(str,map.get(str)-1);
        }
        return res;
    }


    //odd divisor sum
    public int getSum(int num){
        int sum = 0;
        for(int i=1;i<=num/i;++i){
            if(num%i==0){
                if((i&0x1)!=0)
                    sum+=i;
                int complement = num/i;
                if(complement==i)
                    continue;
                if((complement&0x1)!=0)
                    sum+=complement;
            }
        }
        return sum;
    }
    public int getOddDivsiorSum(int[]nums){
        int res = 0;
        Map<Integer,Integer>map=new HashMap<>();
        for(int x:nums){
            if(x<=0)
                continue;
            while((x&0x1)==0)
                x>>=1;
            if(map.containsKey(x))
                res+=map.get(x);
            else{
                int sum = getSum(x);
                map.put(x,sum);
                res+=sum;
            }
        }
        return res;
    }

    /*
    "a" must only be followed by "e".
    "e" must only be followed by "a" or "i".
    "i" must only be followed by "a", "e", "o", or "u".
    "o" must only be followed by "i" or "u".
    "u" must only be followed by "a".
     */
    public int magicString(int n){
        if(n==0)
            return 0;
        if(n==1)
            return 5;
        long [][]dp=new long [n+1][5];
        dp[1][0]=dp[1][1]=dp[1][2]=dp[1][3]=dp[1][4]=1l;
        for(int i=2;i<=n;++i){
            dp[i][0]=(dp[i-1][4]%1000000001+dp[i-1][2]%1000000001+dp[i-1][1]%1000000001)%1000000001;
            dp[i][1]=(dp[i-1][0]%1000000001+dp[i-1][2]%1000000001)%1000000001;
            dp[i][2]=(dp[i-1][1]%1000000001+dp[i-1][3]%1000000001)%1000000001;
            dp[i][3]=dp[i-1][2]%1000000001;
            dp[i][4]=(dp[i-1][2]%1000000001+dp[i-1][3]%1000000001)%1000000001;

        }
        //声明成long
        return (int)((dp[n][0]%1000000001+dp[n][1]%1000000001+dp[n][2]%1000000001+dp[n][3]%1000000001+dp[n][4]%1000000001)%1000000001);
    }

    //array reduction: http://www.1point3acres.com/bbs/thread-201393-1-1.html
    //最少插入（任何位置）多少个字母 能使得输入的字符串变成回文。
    //http://www.geeksforgeeks.org/dynamic-programming-set-28-minimum-insertions-to-form-a-palindrome/
    ////PalindromeCount，找一共有多少substring是palindrome// 2维dp pure storage
    //http://www.geeksforgeeks.org/longest-repeating-and-non-overlapping-substring/
    ///Longest repeating and non-overlapping substring




    //if a string contents with [*] {*} (*) pattern are valid , Given string s, find out s is valid or not

    /*


     */
    //all combinations of a string






















}
