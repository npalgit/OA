package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class Bus extends Vehicle {
    public Bus(){
        spotsNeeded = 5;
        size = VehicleSize.Large;
    }

    //checks if the spot is a large, doesn't check num of spots;
    public boolean canFitInSpot( ParkingSpot spot){
        return spot.getSpotSize()==VehicleSize.Large;
    }

    public void print(){
        System.out.println("B");
    }

}
