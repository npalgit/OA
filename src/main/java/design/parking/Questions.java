package design.parking;

import java.util.Random;

/**
 * Created by tao on 10/24/17.
 */
public class Questions {

    public static Random random =new Random();

    public Questions(){
        //random = new Random();
    }
    public static void main(String[]args){
        ParkingLot lot = new ParkingLot();

        Vehicle v = null;

        while(v==null||lot.parkVehicle(v)){
            lot.print();
            int r = random.nextInt(10);
            if(r<2)
                v = new Bus();
            else if(r<4)
                v = new Motorcycle();
            else
                v = new Car();
            System.out.println("\n parking a ");
            v.print();
            System.out.println("");
        }
        System.out.println("parking failed, final state:");
        lot.print();
    }
}
