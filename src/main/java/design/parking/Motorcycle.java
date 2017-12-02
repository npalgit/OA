package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class Motorcycle extends Vehicle {
    public Motorcycle(){
        spotsNeeded = 1;
        size = VehicleSize.Motorcycle;
    }

    public boolean canFitInSpot(ParkingSpot spot){
        return true;
    }

    public void print(){
        System.out.println("M");
    }
}
