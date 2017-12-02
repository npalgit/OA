package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class Car extends Vehicle {
    public Car(){
        spotsNeeded = 1;
        size = VehicleSize.Compact;
    }

    public boolean canFitInSpot(ParkingSpot spot){
        return spot.getSpotSize()==VehicleSize.Compact||spot.getSpotSize() == VehicleSize.Large;
    }

    public void print(){
        System.out.println("C");
    }
}
