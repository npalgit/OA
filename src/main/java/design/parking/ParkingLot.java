package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class ParkingLot {

    private Level[] levels;
    private final int NUM_LEVELS = 5;

    public ParkingLot(){
        levels = new Level[NUM_LEVELS];
        for(int i=0;i<NUM_LEVELS;++i)
            levels[i] =  new Level(i,30);
    }


    //park the vehicle in a spot (or multiple spots), return false if failed
    public boolean parkVehicle(Vehicle vehicle){
        for(int i=0;i<levels.length;++i){
            if(levels[i].parkVehicle(vehicle))
                return true;
        }
        return false;
    }
    public void print(){
        for(int i=0;i<levels.length;++i){
            System.out.println("level "+i+": ");
            levels[i].print();
            System.out.println("");
        }
        System.out.println("");
    }
}
