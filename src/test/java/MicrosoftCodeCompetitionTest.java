import org.junit.Before;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/7/17.
 */
public class MicrosoftCodeCompetitionTest {


    MicrosoftCodeCompetition micro=null;
    @Before
    public void setup(){
         micro = new MicrosoftCodeCompetition();
    }



    @Test
    public void test(){
        try{
            Scanner in = new Scanner(new FileReader("/Users/tao/Downloads/input.txt"));
            FileWriter fw = null;
            BufferedWriter bw = null;
            fw = new FileWriter("output.txt");
            bw = new BufferedWriter(fw);
            while(in.hasNext()) {
                String date = in.nextLine();
                String[]res=new String[3];
                String []args = date.split("\\s");
                StringBuilder ssb = new StringBuilder(args[2]);
                //2017-09-04 yyyy-mm-dd mm*yyyy*dd
                int ind = args[1].indexOf("yyyy");
                String yy ="";
                if(ind!=-1){
                    yy = args[0].substring(ind,ind+4);
                    //while
                    int indx=0;
                    while(indx<args[2].length()){
                        indx = ssb.toString().indexOf("yyyy",indx);
                        if(indx==-1)
                            break;
                        for(int j=indx;j<indx+4;++j){
                            ssb.setCharAt(j,yy.charAt(j-indx));
                        }
                        indx+=4;
                    }
                }



                ind = args[1].indexOf("mm");
                yy ="";
                if(ind!=-1){
                    yy = args[0].substring(ind,ind+2);
                    //while
                    int indx=0;
                    while(indx<args[2].length()){
                        indx = ssb.toString().indexOf("mm",indx);
                        if(indx==-1)
                            break;
                        for(int j=indx;j<indx+2;++j){
                            ssb.setCharAt(j,yy.charAt(j-indx));
                        }
                        indx+=2;
                    }
                }

                ind = args[1].indexOf("dd");
                yy ="";
                if(ind!=-1){
                    yy = args[0].substring(ind,ind+2);
                    //while
                    int indx=0;
                    while(indx<args[2].length()){
                        indx = ssb.toString().indexOf("dd",indx);
                        if(indx==-1)
                            break;
                        for(int j=indx;j<indx+2;++j){
                            ssb.setCharAt(j,yy.charAt(j-indx));
                        }
                        indx+=2;
                    }
                }

                try {
                    System.out.println(ssb.toString());
                    bw.write(ssb.toString());
                    bw.newLine();
                }catch (Exception e){
                    System.out.println("error");

                }
            }
            bw.close();
            in.close();
        }catch (Exception e){

        }
    }

}