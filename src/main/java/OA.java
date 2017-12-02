import java.util.*;

/**
 * Created by tao on 8/12/17.
 */
public class OA {


    public void printPrime(int n){
        boolean []isPrime = new boolean[n+1];
        Arrays.fill(isPrime,true);
        for(int i=2;i<=n;++i){
            if(isPrime[i]){
                for(int j=2;j<=n/i;++j){
                    isPrime[j*i]=false;
                }
            }
        }
        for(int i=2;i<=n;++i){
            if(isPrime[i])
                System.out.println(i);
        }
    }

    public int getPowerNumber(int n){
        int []indexs=new int[46341];
        Arrays.fill(indexs,2);
        int res =4;
        while(n>0){
            n--;
            int minVal=Integer.MAX_VALUE;
            for(int i=2;i<=46340;++i){
                minVal=Math.min(minVal,(int)Math.pow(i,indexs[i]));
            }
            res=minVal;
            //System.out.println(res);
            for(int i=2;i<=46340;++i){
                if(minVal>=(int)Math.pow(i,indexs[i]))
                    indexs[i]++;
            }

        }
        return res;
    }


    public List<Integer> getPowerNumber1(int n){
        Set<Integer> set=new HashSet<>();
        for(int i=2;i<=46340;++i){
            long num =(long)i;
            for(int j=1;j<=31;++j){
                num*=(long)i;
                if(num>Integer.MAX_VALUE)
                    break;
                set.add((int)num);
            }
        }
        List<Integer>res=new ArrayList<>(set);
        Collections.sort(res);
        System.out.println(res.size());
        //System.out.println(res);
        return res;
        //return res.get(n-1);
        // /* this is a // // come */
        /* dafdsa *** */
        /* dafdfa /* dafd */
        /* afadd // adfad */
    }
}
